"""
The learning-rate schedule and the validation-error measure of a training run. The full pipeline
(data preparation, training, MLflow logging) is exercised end to end by scripts/train.py.
"""

import pytest
import torch

from src.common.config_schema import TrainConfig
from src.training.pipeline import WARMUP_START_FACTOR, TrainingOutcome, build_scheduler
from src.training.trainer import FitResult


def learning_rates(**overrides) -> list[float]:
    """Learning rate used at each epoch."""
    cfg = TrainConfig(experiment_name="e", split="s", model="m", epochs=10, batch_size=8, learning_rate=1e-3, **overrides)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=cfg.learning_rate)
    scheduler = build_scheduler(optimizer, cfg)
    rates = []
    for _ in range(cfg.epochs):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return rates


def test_without_warmup_the_schedule_is_the_plain_cosine():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    expected = []
    for _ in range(10):
        expected.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        cosine.step()
    assert learning_rates() == pytest.approx(expected)


def test_warmup_rises_linearly_then_anneals():
    rates = learning_rates(warmup_epochs=3)
    assert rates[:4] == pytest.approx([WARMUP_START_FACTOR * 1e-3, 4e-4, 7e-4, 1e-3])
    assert all(later < earlier for earlier, later in zip(rates[3:], rates[4:], strict=False))


def test_warmup_then_constant_without_cosine():
    assert learning_rates(warmup_epochs=2, scheduler="none")[2:] == pytest.approx([1e-3] * 8)


def test_no_schedule_at_all():
    assert learning_rates(scheduler="none") == pytest.approx([1e-3] * 10)


def test_mean_val_errors_over_the_last_epochs():
    fit_result = FitResult(best_epoch=1, best_score=0.9, best_metrics={}, final_score=0.9, final_metrics={}, epochs_run=3, stopped_early=False)
    history = [{"val_exact_match": value} for value in (0.5, 0.98, 0.96)]
    outcome = TrainingOutcome(run_id="r", fit=fit_result, history=history, num_parameters=1, flops=1, val_samples=100)
    assert outcome.mean_val_errors(last_k=2) == pytest.approx(3.0)
    assert outcome.mean_val_errors(last_k=1) == pytest.approx(4.0)
