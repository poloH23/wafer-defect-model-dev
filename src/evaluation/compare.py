"""
Aggregate finished MLflow training runs per experiment config, over seeds.

A single run is never enough to compare two configurations: seed-to-seed variation alone moves the
scores about as much as the differences being measured. Each configuration is therefore summarized
as mean ± standard deviation over its runs, each run being represented by the mean of its last
few epochs, which is steadier than any single epoch.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

# Derived metric: exact-match rate turned into a number of misclassified validation samples
ERRORS_METRIC = "val_errors"
# Value of the overrides tag for runs that use their config as is
NO_OVERRIDES = "none"


def normalize_config_path(path: str | Path) -> str:
    """Same form as the config_file tag written by scripts/train.py (e.g. './conf/x.yaml' -> 'conf/x.yaml')."""
    return Path(path).as_posix()


def find_runs(experiment_name: str, config_file: str | Path) -> list[Run]:
    """
    Finished runs of one experiment config made with the current protocol (checkpoint 'last').
    Runs from the earlier protocol (best checkpoint with early stopping) and failed runs are left out:
    their scores are not comparable.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment '{experiment_name}' not found")
    filter_string = f"tags.config_file = '{normalize_config_path(config_file)}' and attributes.status = 'FINISHED' and params.checkpoint = 'last'"
    return mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string=filter_string, output_format="list")


def tail_mean(history: list[tuple[int, float]], last_k: int) -> float:
    """Mean of a metric over its last 'last_k' logged steps (epochs). NaN if the metric was never logged."""
    if not history:
        return math.nan
    values = [value for _, value in sorted(history)][-last_k:]
    return float(np.mean(values))


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation; the deviation is NaN for a single value."""
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else math.nan


def group_by_overrides(runs: list[Run]) -> dict[str, list[Run]]:
    """
    Split the runs of one config by their command-line overrides (e.g. a learning rate), so that
    only repetitions of the same configuration, differing by seed alone, are averaged together.
    Runs made before the overrides tag existed had no overrides.
    """
    groups: dict[str, list[Run]] = {}
    for run in runs:
        groups.setdefault(run.data.tags.get("overrides", NO_OVERRIDES), []).append(run)
    return dict(sorted(groups.items()))


def check_same_split(runs: list[Run]) -> str | None:
    """Every compared run must use the same split: scores on different validation sets are not comparable."""
    digests = {run.data.params.get("split_digest") for run in runs}
    if len(digests) != 1:
        raise ValueError(f"The runs use {len(digests)} different splits; compare runs made on the same split only")
    return digests.pop()


@dataclass
class ConfigSummary:
    config_file: str
    overrides: str
    run_ids: list[str]
    seeds: list[int]
    # metric -> (mean, std) over runs
    metrics: dict[str, tuple[float, float]]
    # Model size and compute per image, identical for every run of the same architecture
    num_parameters: int | None = None
    flops: int | None = None

    @property
    def label(self) -> str:
        name = f"{Path(self.config_file).stem}"
        return name if self.overrides == NO_OVERRIDES else f"{name} ({self.overrides})"

    @property
    def has_duplicate_seeds(self) -> bool:
        return len(set(self.seeds)) != len(self.seeds)


def summarize_runs(
    config_file: str | Path,
    runs: list[Run],
    metrics: list[str],
    last_k: int,
    client: MlflowClient,
    overrides: str = NO_OVERRIDES,
) -> ConfigSummary:
    """
    Per run, the mean of each metric over its last 'last_k' epochs; then mean ± std over runs.
    If val_exact_match is requested and the runs logged val_samples, val_errors is added as well.
    """
    per_run: dict[str, list[float]] = {name: [] for name in metrics}
    errors: list[float] = []
    for run in runs:
        for name in metrics:
            history = [(m.step, m.value) for m in client.get_metric_history(run.info.run_id, name)]
            per_run[name].append(tail_mean(history, last_k))
        if "val_exact_match" in metrics and "val_samples" in run.data.params:
            errors.append((1.0 - per_run["val_exact_match"][-1]) * int(run.data.params["val_samples"]))

    summary = {name: mean_std(values) for name, values in per_run.items()}
    if errors and len(errors) == len(runs):
        summary = {ERRORS_METRIC: mean_std(errors), **summary}
    params = runs[0].data.params
    return ConfigSummary(
        config_file=normalize_config_path(config_file),
        overrides=overrides,
        run_ids=[run.info.run_id for run in runs],
        seeds=sorted(int(run.data.params["seed"]) for run in runs),
        metrics=summary,
        num_parameters=int(params["num_parameters"]) if "num_parameters" in params else None,
        flops=int(params["flops"]) if "flops" in params else None,
    )


def format_mean_std(name: str, mean: float, std: float) -> str:
    digits = 1 if name == ERRORS_METRIC else 4
    return f"{mean:.{digits}f}" if math.isnan(std) else f"{mean:.{digits}f} ± {std:.{digits}f}"


def format_count(value: int | None, unit: float, digits: int) -> str:
    return "-" if value is None else f"{value / unit:.{digits}f}"


def to_markdown(summaries: list[ConfigSummary], last_k: int) -> str:
    """Comparison table: one row per configuration, with its size and compute, then one column per metric."""
    metric_names = list(summaries[0].metrics)
    lines = [
        f"Mean of the last {last_k} epochs per run; mean ± standard deviation over runs (seeds). FLOPs per image.",
        "",
        "| Config | Seeds | Params (M) | MFLOPs | " + " | ".join(metric_names) + " |",
        "|---|---|---|---|" + "---|" * len(metric_names),
    ]
    for s in summaries:
        cells = [format_mean_std(name, *s.metrics[name]) for name in metric_names]
        size = f"{format_count(s.num_parameters, 1e6, 2)} | {format_count(s.flops, 1e6, 0)}"
        lines.append(f"| {s.label} | {', '.join(map(str, s.seeds))} | {size} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
