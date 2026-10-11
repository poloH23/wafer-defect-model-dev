"""
Check, by the classes' own definitions, whether two edge defect classes are separable in the data,
and where a run's confusions between them fall. With the ddefect: Edge_Loc (a defect on part of the
dege) and Edge_Ring (a defect along the whole edge).

If the samples the model get wrong have the edge converage typicall of the class it predicted, the
labels themselves are ambiguous for those samples, and no model can be expected to get the them right.

Usage (after scripts/analyze_errors.py has been run on the same run):
    python scripts/analyze_edge_coverage.py --run-id <MLflow run id>

The label distributions are described on the training and validattion splits only;
the test split is not touched.
"""

import argparse
import json
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from src.evaluation.edge_coverage import edge_coverage, edge_coverage_batch, percentile_rank, separability_auc
from src.evaluation.run_context import load_clean_arrays, load_run_context

from src.data.split import load_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edge coverage analysis of two edge defect classes.")
    parser.add_argument("--run-id", required=True, help="MLflow run id, already analyzed by scripts/analyze_errors.py.")
    parser.add_argument("--partial-class", default="Edge_Loc", help="Class defined as a defect on part of the edge.")
    parser.add_argument("--full-class", default="Edge_Ring", help="Class defined as a defect along the whole edge.")
    parser.add_argument("--band-width", type=int, default=2, help="Width in pixels of the edge band.")
    parser.add_argument("--sectors", type=int, default=36, help="Number of angular sectors around the wafer.")
    parser.add_argument("--sector-threshold", type=float, default=0.5, help="Share of failed dies for a sector to count as defective.")
    return parser.parse_args()


def describe(values: np.ndarray) -> dict[str, float]:
    values = values[~np.isnan(values)]
    p5, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    return {"count": int(values.size), "p5": p5, "p25": p25, "median": p50, "p75": p75, "p95": p95}


def swap_records(records: list[dict[str, Any]], missed: str, predicted: str) -> list[dict[str, Any]]:
    """Misclassified samples whose label `missed` was replaced by `predicted`."""
    return [r for r in records if missed in r["true"] and predicted not in r["true"] and predicted in r["predicted"] and missed not in r["predicted"]]


def main() -> None:
    args = parse_args()
    params = {"band_width": args.band_width, "num_sectors": args.sectors, "sector_threshold": args.sector_threshold}
    with tempfile.TemporaryDirectory() as tmp_dir:
        ctx = load_run_context(args.run_id, tmp_dir)
        try:
            errors_path = mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path="analysis/val_errors.json", dst_path=tmp_dir)
        except Exception as e:
            raise FileNotFoundError(f"Run scripts/analyze_errors.py --run-id {args.run_id} first") from e
        with open(errors_path, encoding="utf-8") as f:
            records = json.load(f)["samples"]

    names = ctx.dataset_cfg.class_names
    partial, full = names.index(args.partial_class), names.index(args.full_class)
    images, labels = load_clean_arrays(ctx)
    split = load_split(ctx.split_cfg.output_dir)
    pool = np.sort(np.concatenate([split.train_idx, split.val_idx]))

    print(f"Computing edge coverage for {pool.size} training and validation samples...")
    coverage = edge_coverage_batch(images[pool], **params)
    pool_labels = labels[pool]
    groups = {
        args.partial_class: coverage[(pool_labels[:, partial] == 1) & (pool_labels[:, full] == 0)],
        args.full_class: coverage[(pool_labels[:, full] == 1) & (pool_labels[:, partial] == 0)],
        "neither": coverage[(pool_labels[:, partial] == 0) & (pool_labels[:, full] == 0)],
    }
    auc = separability_auc(groups[args.partial_class], groups[args.full_class])

    swaps: dict[str, list[dict[str, Any]]] = {}
    for missed, predicted in ((args.partial_class, args.full_class), (args.full_class, args.partial_class)):
        rows = []
        for record in swap_records(records, missed, predicted):
            value = edge_coverage(images[record["index"]], **params)
            rows.append(
                {
                    "index": record["index"],
                    "edge_coverage": value,
                    f"percentile_in_{args.partial_class}": float(percentile_rank(np.array([value]), groups[args.partial_class])[0]),
                    f"percentile_in_{args.full_class}": float(percentile_rank(np.array([value]), groups[args.full_class])[0]),
                    f"p_{args.partial_class}": record["probabilities"][args.partial_class],
                    f"p_{args.full_class}": record["probabilities"][args.full_class],
                }
            )
        swaps[f"{missed}_predicted_as_{predicted}"] = rows

    print(f"\nEdge coverage by label (share of edge sectors with mostly failed dies; {params}):")
    for name, values in groups.items():
        d = describe(values)
        print(f"  {name:<12} n={d['count']:>5} | p5 {d['p5']:.2f}  p25 {d['p25']:.2f}  median {d['median']:.2f}  p75 {d['p75']:.2f}  p95 {d['p95']:.2f}")
    print(f"\nSeparability of {args.partial_class} vs {args.full_class} by coverage alone (ROC AUC): {auc:.3f}")

    # For each confused sample: the share of each class with a coverage at or below its own.
    # E.g. "Edge_Loc 98%" means this sample covers more of the edge than 98% of Edge_Loc samples.
    for key, rows in swaps.items():
        if not rows:
            continue
        print(f"\n{key}: {len(rows)} samples")
        for r in rows:
            print(
                f"  #{r['index']:<6} coverage {r['edge_coverage']:.2f} | "
                f"percentile in {args.partial_class} {r[f'percentile_in_{args.partial_class}']:>5.1f}%, in {args.full_class} {r[f'percentile_in_{args.full_class}']:>5.1f}% | "
                f"p({args.partial_class}) {r[f'p_{args.partial_class}']:.3f}, p({args.full_class}) {r[f'p_{args.full_class}']:.3f}"
            )

    figure, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, args.sectors + 2)
    for name, values in groups.items():
        ax.hist(values[~np.isnan(values)], bins=bins, density=True, histtype="step", linewidth=2, label=f"{name} (n={values.size})")
    for (key, rows), color in zip(swaps.items(), ("tab:red", "tab:purple"), strict=True):
        for k, row in enumerate(rows):
            ax.axvline(row["edge_coverage"], color=color, linestyle="--", alpha=0.7, label=key if k == 0 else None)
    ax.set_xlabel("Edge coverage (share of edge sectors with mostly failed dies)")
    ax.set_ylabel("Density")
    ax.set_title(f"{args.partial_class} vs {args.full_class}: separability AUC {auc:.3f}")
    ax.legend()
    figure.tight_layout()

    with mlflow.start_run(run_id=args.run_id):
        mlflow.log_dict(
            {
                "parameters": {**params, "partial_class": args.partial_class, "full_class": args.full_class},
                "population": "train + val splits",
                "groups": {name: describe(values) for name, values in groups.items()},
                "separability_auc": auc,
                "swaps": swaps,
            },
            "analysis/edge_coverage.json",
        )
        mlflow.log_figure(figure, "analysis/edge_coverage.png")
    plt.close(figure)
    print(f"\nAnalysis attached to run {args.run_id} under analysis/")


if __name__ == "__main__":
    main()
