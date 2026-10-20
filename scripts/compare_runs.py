"""
Compare experiment configs over seeds, from the runs recorded in MLflow.

Usage:
    python scripts/compare_runs.py --configs conf/experiments/wafer_baseline.yaml conf/experiments/wafer_augment.yaml
    python scripts/compare_runs.py --configs ... --output reports/experiments/augmentation_ablation.md

Only finished runs made with checkpoint: last are included, and all of them must use the same split.
A difference between configs smaller than their standard deviations is not evidence of an improvement.
"""

import argparse
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.common.config_schema import TrainConfig, load_config, resolve_path
from src.evaluation.compare import check_same_split, find_runs, group_by_overrides, summarize_runs, to_markdown
from src.training.tracking import resolve_tracking_uri

DEFAULT_METRICS = ["val_exact_match", "val_macro_f1", "val_macro_ap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model from an experiment config.")
    parser.add_argument("--configs", nargs="+", required=True, help="Experiment config paths, as passed to scripts/train.py.")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, help="Per-epoch metrics to compare.")
    parser.add_argument("--last-k", type=int, default=5, help="Number of final epochs averaged per run.")
    parser.add_argument("--output", help="Optional Markdown file for the comparison table, e.g. under reports/experiments/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = {load_config(config, TrainConfig).experiment_name for config in args.configs}
    if len(experiments) != 1:
        raise ValueError(f"The configs belong to different MLflow experiments: {sorted(experiments)}")
    experiment_name = experiments.pop()

    mlflow.set_tracking_uri(resolve_tracking_uri())
    client = MlflowClient()
    runs_by_config = {config: find_runs(experiment_name, config) for config in args.configs}
    for config, runs in runs_by_config.items():
        if not runs:
            raise ValueError(f"No finished run with checkpoint 'last' for {config}")
    check_same_split([run for runs in runs_by_config.values() for run in runs])

    # Within a config, runs with different overrides (e.g. learning rates) are separate rows
    summaries = [
        summarize_runs(config, group, args.metrics, args.last_k, client, overrides=overrides)
        for config, runs in runs_by_config.items()
        for overrides, group in group_by_overrides(runs).items()
    ]
    for summary in summaries:
        if summary.has_duplicate_seeds:
            print(f"WARNING: {summary.label} has several runs with the same seed: {summary.seeds}")
        if len(summary.seeds) < 2:
            print(f"WARNING: {summary.label} has a single run, so its variance is unknown")

    table = to_markdown(summaries, args.last_k)
    print(table)
    if args.output:
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table, encoding="utf-8")
        print(f"Saved: {Path(args.output).as_posix()}")


if __name__ == "__main__":
    main()
