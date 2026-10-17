"""
Train a multi-label classifier from an experiment config and record the run in MLflow.

Usage:
    python scripts/train.py --config conf/experiments/wafer_baseline.yaml --sanity-check
    python scripts/train.py --config conf/experiments/wafer_baseline.yaml

The test split is never touched here: model selection uses the validation split only. The test set
is evaluated exactly once, at the very end of the project, by a separate evaluation step.
"""

import argparse
import sys
from pathlib import Path

from src.common.config_schema import TrainConfig, load_config, load_model_config, resolve_path
from src.training.pipeline import build_components, prepare_data, run_sanity_check, train_and_log
from src.training.trainer import resolve_device


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


def main() -> None:
    args = parse_args()
    train_cfg, overrides = apply_overrides(load_config(args.config, TrainConfig), args)
    model_cfg = load_model_config(train_cfg.model)
    device = resolve_device()
    data = prepare_data(train_cfg.split)

    if args.sanity_check:
        passed = run_sanity_check(build_components(train_cfg, model_cfg, data, device), data, train_cfg, device)
        sys.exit(0 if passed else 1)

    run_name = Path(args.config).stem
    if "learning_rate" in overrides:
        run_name += f"_lr{overrides['learning_rate']:g}"
    if args.seed is not None:
        run_name += f"_seed{args.seed}"
    overrides_tag = ",".join(f"{key}={value}" for key, value in sorted(overrides.items())) or "none"

    train_and_log(
        train_cfg,
        model_cfg,
        data,
        device,
        run_name=run_name,
        tags={"config_file": Path(args.config).as_posix(), "overrides": overrides_tag},
        config_files=[resolve_path(args.config)],
    )


if __name__ == "__main__":
    main()
