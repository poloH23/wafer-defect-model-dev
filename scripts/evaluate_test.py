"""
The single evaluation on the test split, following docs/model_selection.md ("Test evaluation").

Usage:
    python scripts/evaluate_test.py --config conf/evaluation/final_test.yaml

For every candidate and seed: per-class thresholds are tuned on the validation split, then the model
is evaluated on the test split, at the tuned thresholds and at the 0.5 default. Each candidate is
summarized over its seeds, with a bootstrap confidence interval over test samples, and the selected
configuration is compared with each other candidate on identical resamples.

The test split is meant to be evaluated once: the script refuses to run when the report already
exists. Deleting the report to run again is a deliberate act, justified only if the previous run
failed before producing results.
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from mlflow.entities import Run
from torch import nn
from torch.utils.data import DataLoader

from src.common.config_schema import CandidateRef, TestEvaluationConfig, load_config, load_experiment_config, resolve_path
from src.data.dataset import DiscreteMapDataset
from src.data.split import load_split
from src.evaluation.bootstrap import bootstrap_ci, paired_difference, resample_weights, sample_errors
from src.evaluation.compare import check_same_split, find_runs, group_by_overrides
from src.evaluation.run_context import download_checkpoint, load_clean_arrays, load_run_context
from src.evaluation.thresholds import DEFAULT_THRESHOLD, tune_thresholds
from src.models.factory import load_checkpoint
from src.training.metrics import multilabel_metrics
from src.training.tracking import committed_version, git_tags, resolve_tracking_uri
from src.training.trainer import predict, resolve_device

DEFAULT_OUTPUT = "reports/final/test_evaluation.md"
SUMMARY_METRICS = ("exact_match", "macro_f1", "macro_ap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="The single evaluation on the test split.")
    parser.add_argument("--config", required=True, help="Test evaluation config, e.g. conf/evaluation/final_test.yaml.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Markdown report; a JSON file with every number is written next to it.")
    return parser.parse_args()


def label(candidate: CandidateRef) -> str:
    return candidate.experiment if candidate.overrides == "none" else f"{candidate.experiment} ({candidate.overrides})"


def candidate_runs(candidate: CandidateRef, seeds: list[int]) -> dict[int, Run]:
    """The candidate's runs, exactly one per configured seed."""
    experiment_name = load_experiment_config(candidate.experiment).experiment_name
    # The config_file tag that scripts/train.py records for runs of this experiment config
    runs = group_by_overrides(find_runs(experiment_name, f"conf/experiments/{candidate.experiment}.yaml")).get(candidate.overrides, [])
    by_seed: dict[int, list[Run]] = {}
    for run in runs:
        by_seed.setdefault(int(run.data.params["seed"]), []).append(run)
    if sorted(by_seed) != sorted(seeds) or any(len(found) != 1 for found in by_seed.values()):
        raise ValueError(f"{label(candidate)}: expected exactly one finished run per seed {seeds}, found seeds {sorted(by_seed)} with counts {[len(v) for v in by_seed.values()]}")
    return {seed: found[0] for seed, found in by_seed.items()}


def predict_split(
    model: nn.Module, images: np.ndarray, labels: np.ndarray, indices: np.ndarray, num_values: int, encoding: str, batch_size: int, device
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(DiscreteMapDataset(images, labels, indices, num_values, encoding), batch_size=batch_size, shuffle=False)
    # The loss is not used; predict() only needs a loss function to report one
    y_true, y_prob, _ = predict(model, loader, nn.BCEWithLogitsLoss(), device)
    return y_true.astype(int), y_prob


def evaluate_run(run: Run, data: dict[str, Any], device) -> dict[str, Any]:
    """Tune thresholds on validation, evaluate on test, and record the result in the run itself."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ctx = load_run_context(run.info.run_id, tmp_dir)
        model, checkpoint = load_checkpoint(download_checkpoint(ctx, tmp_dir), device)
    names = checkpoint["class_names"]
    common = (data["images"], data["labels"])
    args = (ctx.num_values, checkpoint["encoding"], ctx.train_cfg.batch_size, device)
    val_true, val_prob = predict_split(model, *common, data["split"].val_idx, *args)
    test_true, test_prob = predict_split(model, *common, data["split"].test_idx, *args)

    thresholds = tune_thresholds(val_true, val_prob)
    result = {
        "run_id": run.info.run_id,
        "thresholds": dict(zip(names, thresholds.tolist(), strict=True)),
        "val_errors_default": int(sample_errors(val_true, val_prob >= DEFAULT_THRESHOLD).sum()),
        "tuned": multilabel_metrics(test_true, test_prob, thresholds, names),
        "default": multilabel_metrics(test_true, test_prob, DEFAULT_THRESHOLD, names),
        "errors_tuned": sample_errors(test_true, test_prob >= thresholds),
        "errors_default": sample_errors(test_true, test_prob >= DEFAULT_THRESHOLD),
    }
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_dict(
            {k: v for k, v in result.items() if not k.startswith("errors_")}
            | {"test_errors_tuned": int(result["errors_tuned"].sum()), "test_errors_default": int(result["errors_default"].sum())},
            "test/test_evaluation.json",
        )
        mlflow.set_tag("test_evaluated", "true")
    return result


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)) if array.size > 1 else float("nan")}


def summarize(candidate: CandidateRef, results: dict[int, dict[str, Any]], weights: np.ndarray, level: float) -> dict[str, Any]:
    seeds = sorted(results)
    tuned_errors = np.stack([results[s]["errors_tuned"] for s in seeds])
    return {
        "candidate": label(candidate),
        "seeds": seeds,
        "val_errors_default": mean_std([results[s]["val_errors_default"] for s in seeds]),
        "test_errors_default": mean_std([int(results[s]["errors_default"].sum()) for s in seeds]),
        "test_errors_tuned": mean_std([int(results[s]["errors_tuned"].sum()) for s in seeds]),
        "test_errors_tuned_ci": bootstrap_ci(tuned_errors, weights, level),
        **{f"test_{name}": mean_std([results[s]["tuned"][name] for s in seeds]) for name in SUMMARY_METRICS},
    }


def fmt(stats: dict[str, float], digits: int = 1) -> str:
    return f"{stats['mean']:.{digits}f}" if np.isnan(stats["std"]) else f"{stats['mean']:.{digits}f} ± {stats['std']:.{digits}f}"


def to_markdown(
    cfg: TestEvaluationConfig, summaries: list[dict[str, Any]], comparisons: list[dict[str, Any]], selected_results: dict[int, dict[str, Any]], n_test: int, code_version: str
) -> str:
    level = round(cfg.confidence_level * 100)
    lines = [
        "# Test evaluation",
        "",
        f"The single evaluation on the test split ({n_test:,} samples), following docs/model_selection.md. Thresholds are tuned per class "
        f"on the validation split for each seed. Errors: samples with any label wrong. Mean ± standard deviation over seeds {cfg.seeds}; "
        f"{level}% intervals from {cfg.bootstrap_resamples:,} bootstrap resamples of test samples.",
        "",
        f"Code and protocol: commit '{code_version}'.",
        "",
        f"**Selected for all deployment targets: {label(cfg.selected)}.** The other candidates are evaluated for transparency only: their results do not change the selection.",
        "",
        "## All candidates",
        "",
        f"| Candidate | Val errors (0.5) | Test errors (0.5) | Test errors (tuned) | {level}% interval | Exact match | Macro F1 | Macro AP |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        ci = s["test_errors_tuned_ci"]
        name = f"**{s['candidate']}** (selected)" if s["candidate"] == label(cfg.selected) else s["candidate"]
        lines.append(
            f"| {name} | {fmt(s['val_errors_default'])} | {fmt(s['test_errors_default'])} | {fmt(s['test_errors_tuned'])} | "
            f"{ci['ci_low']:.1f} to {ci['ci_high']:.1f} | {fmt(s['test_exact_match'], 4)} | {fmt(s['test_macro_f1'], 4)} | {fmt(s['test_macro_ap'], 4)} |"
        )
    lines += [
        "",
        "## Selected configuration against each other candidate",
        "",
        "Seed-averaged test errors (tuned thresholds) of the selected configuration minus those of the other candidate, on identical resamples. "
        "Negative: the selected configuration makes fewer errors. An interval containing 0 means no detectable difference.",
        "",
        f"| Other candidate | Difference | {level}% interval | Resamples where the selected one makes fewer errors |",
        "|---|---|---|---|",
    ]
    for c in comparisons:
        lines.append(f"| {c['other']} | {c['difference']:+.1f} | {c['ci_low']:+.1f} to {c['ci_high']:+.1f} | {c['share_a_fewer']:.0%} |")

    names = list(selected_results[cfg.seeds[0]]["thresholds"])
    seeds = sorted(selected_results)
    lines += [
        "",
        f"## Selected configuration per class (tuned thresholds, mean over seeds {seeds})",
        "",
        "| Class | Test support | Precision | Recall | F1 | AP | Thresholds (per seed) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in names:
        per_seed = [selected_results[s]["tuned"]["per_class"][name] for s in seeds]
        mean = {key: float(np.mean([p[key] for p in per_seed])) for key in ("precision", "recall", "f1", "ap")}
        thresholds = ", ".join(f"{selected_results[s]['thresholds'][name]:.2f}" for s in seeds)
        ap = "–" if np.isnan(mean["ap"]) else f"{mean['ap']:.4f}"  # AP is undefined without positive test samples
        lines.append(f"| {name} | {per_seed[0]['support']} | {mean['precision']:.4f} | {mean['recall']:.4f} | {mean['f1']:.4f} | {ap} | {thresholds} |")
    deliverable = selected_results[cfg.deliverable_seed]
    lines += ["", f"**Deliverable**: seed {cfg.deliverable_seed} of the selected configuration (MLflow run {deliverable['run_id']}), with its tuned thresholds."]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output = resolve_path(args.output)
    if output.exists():
        raise FileExistsError(f"{args.output} exists: the test split has already been evaluated (see the docstring before deleting it)")
    # Checked before any test sample is read: the report must point to the exact code and protocol
    code_version = committed_version(git_tags())
    cfg = load_config(args.config, TestEvaluationConfig)
    mlflow.set_tracking_uri(resolve_tracking_uri())
    device = resolve_device()

    candidates = [cfg.selected, *cfg.others]
    runs = {label(c): candidate_runs(c, cfg.seeds) for c in candidates}
    check_same_split([run for by_seed in runs.values() for run in by_seed.values()])

    # Same data and split for every run (checked above): load and clean them once
    first_run = next(iter(runs[label(cfg.selected)].values()))
    with tempfile.TemporaryDirectory() as tmp_dir:
        ctx = load_run_context(first_run.info.run_id, tmp_dir)
    images, labels = load_clean_arrays(ctx)
    data = {"images": images, "labels": labels, "split": load_split(ctx.split_cfg.output_dir)}
    n_test = len(data["split"].test_idx)
    weights = resample_weights(n_test, cfg.bootstrap_resamples, cfg.bootstrap_seed)

    results: dict[str, dict[int, dict[str, Any]]] = {}
    for candidate in candidates:
        print(f"Evaluating {label(candidate)}...")
        results[label(candidate)] = {seed: evaluate_run(run, data, device) for seed, run in runs[label(candidate)].items()}

    summaries = [summarize(c, results[label(c)], weights, cfg.confidence_level) for c in candidates]
    selected_errors = np.stack([r["errors_tuned"] for _, r in sorted(results[label(cfg.selected)].items())])
    comparisons = [
        {"other": label(c), **paired_difference(selected_errors, np.stack([r["errors_tuned"] for _, r in sorted(results[label(c)].items())]), weights, cfg.confidence_level)}
        for c in cfg.others
    ]

    report = to_markdown(cfg, summaries, comparisons, results[label(cfg.selected)], n_test, code_version)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    details = {
        "git_commit": code_version,
        "config": cfg.model_dump(mode="json"),
        "n_test": n_test,
        "summaries": summaries,
        "comparisons": comparisons,
        "runs": {name: {seed: {k: v for k, v in r.items() if not k.startswith("errors_")} for seed, r in by_seed.items()} for name, by_seed in results.items()},
    }
    output.with_suffix(".json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(report)
    print(f"Saved: {Path(args.output).as_posix()} and {Path(args.output).with_suffix('.json').as_posix()}")


if __name__ == "__main__":
    main()
