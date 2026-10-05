# Training and eveluation loops for multi-label classification (one sigmoid per class)
# Delibrately free of MLFlow: per-epoch metrics are reported through an optional callback,
# so the loop can be tested and reused without and tracking backend.
# The training script simply passes mlflow.log_metrics as that callback.

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.metrics import flatten_metrics, multilabel_metrics

# Called once per epoch with (metrics, epoch); matches the signature of mlflow.log_metrics (metrics, step)
EpochCallback = Callable[[dict[str, float], int], Any]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (torch.manual_seed also seeds every CUDA device)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """One pass over the training data. Returns the sample-weighted mean loss."""
    model.train()
    total_loss, total_samples = 0.0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    return total_loss / total_samples


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    """Targets and sigmoid probabilities for a whole loader, plus the sample-weighted mean loss."""
    model.eval()
    all_targets, all_probs = [], []
    total_loss, total_samples = 0.0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        total_loss += loss_fn(logits, targets).item() * images.size(0)
        total_samples += images.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    return np.concatenate(all_targets), np.concatenate(all_probs), total_loss / total_samples


@dataclass
class FitResult:
    best_epoch: int
    best_score: float
    # Full multilabel_metrics output (including pre-class results) of the best epoch
    best_metrics: dict[str, Any]
    epochs_run: int
    stopped_early: bool


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    class_names: list[str],
    monitor: str = "val_macro_ap",
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    early_stopping_patience: int | None = None,
    on_epoch_end: EpochCallback | None = None,
) -> FitResult:
    """
    Train for up to 'epochs', evaluate on the validation set after every epoch, and keep the weights
    of the epoch with the highest 'monitor' value. On return the model holds those best weights.
    """
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch, best_metrics = 0, {}
    stale_epoch, epoch, stopped_early = 0, 0, False

    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        y_true, y_prob, val_loss = predict(model, val_loader, loss_fn, device)
        metrics = multilabel_metrics(y_true, y_prob, class_names=class_names)

        record = {"train_loss": train_loss, "val_loss": val_loss, "lr": lr, **flatten_metrics(metrics, prefix="val")}
        if monitor not in record:
            raise KeyError(f"monitor '{monitor}' is not a logged metric; available metrics: {sorted(record)}")
        if on_epoch_end is not None:
            on_epoch_end(record, epoch)

        score = record[monitor]
        improved = score > best_score
        if improved:
            best_score, best_epoch, best_metrics = score, epoch, metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epoch = 0
        else:
            stale_epoch += 1
        print(f"Epoch {epoch:>3}/{epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | {monitor} {score:.4f} {' *' if improved else ''}")

        if scheduler is not None:
            scheduler.step()
        if early_stopping_patience is not None and stale_epoch >= early_stopping_patience:
            print(f"Early stopping: no improvement in {monitor} for {early_stopping_patience} epochs")
            stopped_early = True
            break

    if best_state is None:
        raise RuntimeError("No epoch produce a valid '{monitor}' score.")
    model.load_state_dict(best_state)
    return FitResult(best_epoch=best_epoch, best_score=best_score, best_metrics=best_metrics, epochs_run=epoch, stopped_early=stopped_early)


def overfit_single_batch(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps: int = 200,
) -> dict[str, float]:
    """
    Sanity check before real training: train repeatedly on ONE small batch.

    A working pipeline drives the loss tward 0 and fits every label of that batch.
    If it cannot, something is broken (labels misaligned with images, wrong target dtype, forzen parameters,
    a learning rate far off, etc.), and a full training run will not run successfully.
    """
    images, targets = images.to(device), targets.to(device)
    model.train()
    with torch.no_grad():
        initial_loss = loss_fn(model(images), targets).item()

    final_loss = initial_loss
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), targets)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    model.eval()
    with torch.no_grad():
        predictions = (torch.sigmoid(model(images)) >= 0.5).float()
    exact_match = (predictions == targets).all(dim=1).float().mean().item()
    return {"initial_loss": initial_loss, "final_loss": final_loss, "exact_match": exact_match}
