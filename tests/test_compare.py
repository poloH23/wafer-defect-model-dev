import math
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from src.evaluation.compare import (
    ERRORS_METRIC,
    check_same_split,
    find_runs,
    group_by_overrides,
    mean_std,
    normalize_config_path,
    summarize_runs,
    tail_mean,
    to_markdown,
)
from src.training import tracking
from src.training.tracking import setup_experiment

EXPERIMENT = "compare_test"
BASELINE = "conf/experiments/baseline.yaml"
AUGMENT = "conf/experiments/augment.yaml"

# ================= fixture: synthesised data =================


@pytest.fixture
def tracking_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(tracking, "get_project_root", lambda: tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    setup_experiment(EXPERIMENT)
    return tmp_path


def log_run(
    config: str,
    seed: int,
    exact_match: list[float],
    checkpoint: str = "last",
    split_digest: str = "d1",
    status: str = "FINISHED",
    overrides: str | None = None,
) -> str:
    run = mlflow.start_run(run_name=f"{Path(config).stem}_seed{seed}")
    mlflow.set_tags({"config_file": config, **({"overrides": overrides} if overrides else {})})
    mlflow.log_params({"checkpoint": checkpoint, "seed": seed, "split_digest": split_digest, "val_samples": 100, "num_parameters": 11_170_000, "flops": 3_600_000_000})
    for epoch, value in enumerate(exact_match, start=1):
        mlflow.log_metrics({"val_exact_match": value, "val_macro_ap": 0.99}, step=epoch)
    mlflow.end_run(status=status)
    return run.info.run_id


def test_tail_mean_uses_the_last_epochs_in_step_order():
    history = [(3, 0.8), (1, 0.2), (2, 0.6), (4, 1.0)]
    assert tail_mean(history, last_k=2) == pytest.approx(0.9)
    assert math.isnan(tail_mean([], last_k=2))


# ================= Test: mean_std =================


def test_mean_std():
    assert mean_std([1.0, 3.0]) == (2.0, pytest.approx(math.sqrt(2)))
    mean, std = mean_std([5.0])
    assert mean == 5.0 and math.isnan(std)


# ================= Test: normalize_config_path =================


def test_normalize_config_path():
    assert normalize_config_path("./conf/experiments/baseline.yaml") == BASELINE


# ================= Test: find_runs =================


def test_find_runs_keeps_only_finished_runs_of_the_current_protocol(tracking_root):
    kept = {log_run(BASELINE, 0, [0.9]), log_run(BASELINE, 1, [0.9])}
    log_run(BASELINE, 2, [0.9], checkpoint="best")  # earlier protocol
    log_run(BASELINE, 3, [0.9], status="FAILED")
    log_run(AUGMENT, 0, [0.9])  # another config
    assert {run.info.run_id for run in find_runs(EXPERIMENT, "./" + BASELINE)} == kept


def test_find_runs_unknown_experiment_raises(tracking_root):
    with pytest.raises(ValueError, match="not found"):
        find_runs("missing_experiment", BASELINE)


# ================= Test: summarize_runs =================


def test_summarize_runs_averages_the_last_epochs_then_the_seeds(tracking_root):
    log_run(BASELINE, 0, [0.5, 0.9, 0.98, 0.96])  # last 2 epochs -> 0.97
    log_run(BASELINE, 1, [0.5, 0.9, 0.94, 0.96])  # last 2 epochs -> 0.95
    summary = summarize_runs(BASELINE, find_runs(EXPERIMENT, BASELINE), ["val_exact_match"], last_k=2, client=MlflowClient())

    assert summary.seeds == [0, 1]
    mean, std = summary.metrics["val_exact_match"]
    assert mean == pytest.approx(0.96)
    assert std == pytest.approx(math.sqrt(2) * 0.01)
    # 3 and 5 misclassified samples out of 100
    assert summary.metrics[ERRORS_METRIC][0] == pytest.approx(4.0)
    assert list(summary.metrics)[0] == ERRORS_METRIC


def test_duplicate_seeds_are_flagged(tracking_root):
    log_run(BASELINE, 0, [0.9])
    log_run(BASELINE, 0, [0.9])
    summary = summarize_runs(BASELINE, find_runs(EXPERIMENT, BASELINE), ["val_exact_match"], last_k=1, client=MlflowClient())
    assert summary.has_duplicate_seeds


# ================= Test: check_same_split =================


def test_runs_on_different_splits_cannot_be_compared(tracking_root):
    log_run(BASELINE, 0, [0.9], split_digest="d1")
    log_run(AUGMENT, 0, [0.9], split_digest="d2")
    runs = find_runs(EXPERIMENT, BASELINE) + find_runs(EXPERIMENT, AUGMENT)
    with pytest.raises(ValueError, match="different splits"):
        check_same_split(runs)


# ================= Test: summarize_runs & find_runs =================


def test_markdown_table(tracking_root):
    log_run(BASELINE, 0, [0.96])
    log_run(BASELINE, 1, [0.98])
    log_run(AUGMENT, 0, [0.97])
    client = MlflowClient()
    summaries = [summarize_runs(c, find_runs(EXPERIMENT, c), ["val_exact_match"], last_k=1, client=client) for c in (BASELINE, AUGMENT)]
    table = to_markdown(summaries, last_k=1)
    assert "| baseline | 0, 1 | 11.17 | 3600 | 3.0 ± 1.4 | 0.9700 ± 0.0141 |" in table
    assert "| augment | 0 | 11.17 | 3600 | 3.0 | 0.9700 |" in table  # a single run has no standard deviation


# ================= Test: group_by_overrides =================


def test_runs_with_different_overrides_are_separate_rows(tracking_root):
    log_run(BASELINE, 0, [0.90])  # older run without the overrides tag
    log_run(BASELINE, 1, [0.92], overrides="none")
    log_run(BASELINE, 0, [0.80], overrides="learning_rate=0.0003")
    groups = group_by_overrides(find_runs(EXPERIMENT, BASELINE))
    assert {key: len(runs) for key, runs in groups.items()} == {"learning_rate=0.0003": 1, "none": 2}

    client = MlflowClient()
    summaries = [summarize_runs(BASELINE, runs, ["val_exact_match"], last_k=1, client=client, overrides=key) for key, runs in groups.items()]
    table = to_markdown(summaries, last_k=1)
    assert "| baseline (learning_rate=0.0003) | 0 |" in table
    assert "| baseline | 0, 1 |" in table
