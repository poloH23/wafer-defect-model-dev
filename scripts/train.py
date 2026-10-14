"""
Train a nulti-label classifier from an experiment config and record the run in MLflow.

Usage:
    python scripts/train.py --config conf/experiments/wafer_baseline.yaml --sanity-check
    python scripts/train.py --config conf/experiments/wafer_baseline.yaml

The test split is never touched here.
"""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.common.config_schema import TrainConfig, load_config, load_dataset_config, load_model_config, load_split_config, resolve_path
from src.common.hashing import file_sha256
from src.data.cleaning import apply_invalid_pixel_policy
from src.data.dataset import DiscreteMapDataset, build_split_datasets, seed_worker
from src.data.io import load_npz_arrays
from src.data.split import META_FILE, load_split, load_split_meta
from src.data.transforms import RandomDihedral, encoded_channels
from src.models.factory import build_model
from src.models.profiling import count_flops, count_parameters
from src.training.class_balance import compute_pos_weight, expected_initial_bce
from src.training.losses import build_loss
from src.training.tracking import flatten_params, git_tags, setup_experiment
from src.training.trainer import fit, overfit_single_batch, resolve_device, set_seed

SANITY_BATCH_SIZE = 32
SANITY_STEPS = 200
# Recorded as a run tag, so analysis scripts find the checkpoint without hard-coding its path
CHECKPOINT_ARTIFACT = "checkpoints/model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model from an experiment config.")
    parser.add_argument("--config", required=True, help="Experiment config path, relative to the project root.")
    parser.add_argument("--sanity-check", action="store_true", help="Only overfit a single batch to verify the pipeline; nothing is logged.")
    parser.add_argument("--seed", type=int, help="Override the config seed, e.g. to measure run-to-run variance. The effective seed is what gets logged.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override the config learning rate, e.g. for a small per-architecture learning rate comparison. "
        "Recorded in the 'overrides' tag, so comparisons keep runs with different learning rates apart.",
    )
    return parser.parse_args()


def apply_overrides(cfg: TrainConfig, args: argparse.Namespace) -> tuple[TrainConfig, dict[str, float]]:
    """
    Apply command-line overrides other than the seed, and return the ones that actually change the config.
    Seeds are not recorded as overrides: runs that differ only by seed are repetitions of one configuration.
    """
    overrides: dict[str, float] = {}
    if args.learning_rate is not None and args.learning_rate != cfg.learning_rate:
        overrides["learning_rate"] = args.learning_rate
    updates: dict[str, float | int] = {**overrides}
    if args.seed is not None:
        updates["seed"] = args.seed
    return cfg.model_copy(update=updates), overrides


def build_loaders(datasets: dict[str, DiscreteMapDataset], cfg: TrainConfig, device: torch.device) -> dict[str, DataLoader]:
    common: dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": cfg.num_workers > 0,
    }
    # A dedicated generator makes the shuffling order reproducible from the config seed
    generator = torch.Generator().manual_seed(cfg.seed)
    return {
        "train": DataLoader(datasets["train"], shuffle=True, generator=generator, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
    }


def run_sanity_check(
    model: nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    num_values: int,
    cfg: TrainConfig,
    pos_weight: np.ndarray | None,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> bool:
    # Random training samples, not the first ones: split indices are sorted and the data file is
    # ordered by class, so the first 32 samples would all share one label combination and
    # "overfitting" them would prove almost nothing.
    batch_idx = np.random.default_rng(cfg.seed).choice(train_idx, size=SANITY_BATCH_SIZE, replace=True)
    batch = DiscreteMapDataset(images, labels, batch_idx, num_values, cfg.encoding, transform=None)
    batch_images = torch.stack([batch[i][0] for i in range(len(batch))])
    batch_targets = torch.stack([batch[i][1] for i in range(len(batch))])

    print(f"Sanity check: overfitting one batch of {SANITY_BATCH_SIZE} training samples for {SANITY_STEPS} steps on {device}")
    result = overfit_single_batch(model, batch_images, batch_targets, loss_fn, optimizer, device, steps=SANITY_STEPS)
    expected = expected_initial_bce(labels[batch_idx], pos_weight)

    print(f"  initial loss : {result['initial_loss']:.4f} (excepted about {expected:.4f} for an untrained model)")
    print(f"  final lose   : {result['final_loss']:.4f}")
    print(f"  exact match  : {result['exact_match']:.1%} of the batch")

    if not 0.5 * expected <= result["initial_loss"] <= 2.0 * expected:
        print(" WARNING: the initial loss is far from the expected value; check the model output and the targets.")
    passed = result["exact_match"] == 1.0 and result["final_loss"] < 0.1 * result["initial_loss"]
    print("PASS" if passed else "FAIL: the model could not fit a single batch; fix this this this before any real training.")
    return passed


def main() -> None:
    args = parse_args()
    train_cfg, overrides = apply_overrides(load_config(args.config, TrainConfig), args)
    split_cfg = load_split_config(train_cfg.split)
    dataset_cfg = load_dataset_config(split_cfg.dataset)
    model_cfg = load_model_config(train_cfg.model)
    if dataset_cfg.num_pixel_values is None:
        raise ValueError("This training script expects discrete-valued maps; set num_pixel_values in the dataset config.")
    num_values = dataset_cfg.num_pixel_values

    set_seed(train_cfg.seed)
    device = resolve_device()

    # Data lineage: the split must have been created from exactly this data file
    split_meta = load_split_meta(split_cfg.output_dir)
    source_sha256 = file_sha256(dataset_cfg.source_path)
    if split_meta.get("source_sha256") != source_sha256:
        raise RuntimeError(f"{dataset_cfg.source_path.name} does not match the file split '{train_cfg.split}' was created from; regenerate the split or restore the data file.")

    images, labels = load_npz_arrays(dataset_cfg.source_path, dataset_cfg.image_key, dataset_cfg.label_key)
    # Validate every image at once, before any model or MLflow run is created
    images, cleaning_report = apply_invalid_pixel_policy(images, num_values, dataset_cfg.invalid_pixel_policy)
    if cleaning_report["invalid_pixels"]:
        print(f"Imputed {cleaning_report['invalid_pixels']} out-of-range pixels in {cleaning_report['invalid_images']} images: {cleaning_report['replaced_with']}")
    split = load_split(split_cfg.output_dir)
    train_transform = RandomDihedral(seed=train_cfg.seed) if train_cfg.augment else None
    datasets = build_split_datasets(images, labels, split, num_values, train_cfg.encoding, train_transform)

    in_channels = encoded_channels(num_values, train_cfg.encoding)
    model = build_model(model_cfg.name, dataset_cfg.num_classes, in_channels, model_cfg.pretrained, model_cfg.small_input_stem).to(device)

    # Class weight come from the training split only
    pos_weight = compute_pos_weight(labels[split.train_idx], train_cfg.pos_weight.mode, train_cfg.pos_weight.max_value)
    loss_fn = build_loss(pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)

    if args.sanity_check:
        passed = run_sanity_check(model, images, labels, split.train_idx, num_values, train_cfg, pos_weight, loss_fn, optimizer, device)
        sys.exit(0 if passed else 1)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.epochs) if train_cfg.scheduler == "cosine" else None
    loaders = build_loaders(datasets, train_cfg, device)

    setup_experiment(train_cfg.experiment_name)
    run_name = Path(args.config).stem
    if "learning_rate" in overrides:
        run_name += f"_lr{overrides['learning_rate']:g}"
    if args.seed is not None:
        run_name += f"_seed{args.seed}"
    overrides_tag = ",".join(f"{key}={value}" for key, value in sorted(overrides.items())) or "none"
    with mlflow.start_run(run_name=run_name), tempfile.TemporaryDirectory() as tmp_dir:
        mlflow.set_tags({**git_tags(), "config_file": Path(args.config).as_posix(), "checkpoint_artifact": CHECKPOINT_ARTIFACT, "overrides": overrides_tag})
        mlflow.log_params(
            {
                **flatten_params(train_cfg.model_dump(mode="json")),
                **flatten_params(model_cfg.model_dump(mode="json"), prefix="model_cfg"),
                "dataset": dataset_cfg.name,
                "split_digest": split_meta["split_digest"],
                "invalid_pixels_imputed": cleaning_report["invalid_pixels"] if cleaning_report["policy"] == "neighbor_mode" else 0,
                "source_sha256": source_sha256,
                "in_channels": in_channels,
                "num_parameters": count_parameters(model),
                # Per image, at the dataset's input size (batch of 1)
                "flops": count_flops(model, (1, in_channels, *images.shape[1:])),
                "device": device.type,
                # Lets comparisons turn exact-match rates into numbers of misclassified samples
                "val_samples": len(datasets["val"]),
            }
        )
        # The exact configs used, so the run can be reproduced even after the YAML files change
        mlflow.log_artifact(str(resolve_path(args.config)), artifact_path="config")
        mlflow.log_artifact(str(split_cfg.output_dir / META_FILE), artifact_path="config")
        if pos_weight is not None:
            mlflow.log_dict(dict(zip(dataset_cfg.class_names, pos_weight.tolist(), strict=True)), "config/pos_weight.json")
        mlflow.log_dict(cleaning_report, "config/pixel_cleaning.json")

        result = fit(
            model,
            loaders["train"],
            loaders["val"],
            loss_fn,
            optimizer,
            epochs=train_cfg.epochs,
            device=device,
            class_names=dataset_cfg.class_names,
            monitor=train_cfg.monitor,
            scheduler=scheduler,
            early_stopping_patience=train_cfg.early_stopping_patience,
            on_epoch_end=mlflow.log_metrics,
            restore_best=train_cfg.checkpoint == "best",
        )

        # The best epoch is always logged for information; the saved model follows train_cfg.checkpoint
        saved_epoch = result.best_epoch if train_cfg.checkpoint == "best" else result.epochs_run
        saved_score = result.best_score if train_cfg.checkpoint == "best" else result.final_score
        mlflow.log_metrics(
            {
                "best_epoch": result.best_epoch,
                f"best_{train_cfg.monitor}": result.best_score,
                f"final_{train_cfg.monitor}": result.final_score,
                "saved_epoch": saved_epoch,
                "epochs_run": result.epochs_run,
            }
        )
        mlflow.log_dict(result.best_metrics, "metrics/best_val_metrics.json")
        mlflow.log_dict(result.final_metrics, "metrics/final_val_metrics.json")

        # Everything needed to rebuild the model for evaluation or ONNX export later
        checkpoint_path = Path(tmp_dir) / Path(CHECKPOINT_ARTIFACT).name
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": model_cfg.model_dump(mode="json"),
                "num_classes": dataset_cfg.num_classes,
                "in_channels": in_channels,
                "encoding": train_cfg.encoding,
                "class_names": dataset_cfg.class_names,
                "epoch": saved_epoch,
                "selection": train_cfg.checkpoint,
            },
            checkpoint_path,
        )
        mlflow.log_artifact(str(checkpoint_path), artifact_path=str(Path(CHECKPOINT_ARTIFACT).parent))
        print(
            f"Saved the {train_cfg.checkpoint} model: epoch {saved_epoch}, {train_cfg.monitor} {saved_score:.4f} "
            f"(best epoch {result.best_epoch}: {result.best_score:.4f}) | run {mlflow.active_run().info.run_id}"
        )


if __name__ == "__main__":
    main()
