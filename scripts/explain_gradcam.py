"""
Grad-CAM analysis of a trained model, normally the deliverable, on the validation split.

Usage:
    python scripts/explain_gradcam.py --run-id <MLflow run id>

Three pieces of evidence, written to reports/final/gradcam.md with figures and logged to the run:

1. Localization, on every correctly classified single-defect sample of the classes that lie in a
   known region (Center, Donut, Edge_Loc, Edge_Ring): the share of heatmap mass inside that region,
   relative to the region's share of the wafer. Compared with the same architecture with random
   weights, whose maps should neither focus nor resemble the trained model's.
2. Example heatmaps for every class, on correctly classified single-defect samples drawn at random
   with a fixed seed, not picked for how they look.
3. The Edge_Loc samples predicted as Edge_Ring, with the heatmaps of both classes and their edge coverage.

The analysis is descriptive and does not change the model. It uses the validation split by default,
so that the test split remains used for its single evaluation only.
"""

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from mlflow.tracking import MlflowClient
from torch import nn
from torch.utils.data import DataLoader

from src.common.config_schema import ModelConfig, resolve_path
from src.data.dataset import DiscreteMapDataset
from src.data.split import load_split
from src.evaluation.edge_coverage import angular_sectors, edge_band, edge_coverage, edge_sector_failures
from src.evaluation.explain import (
    CLASS_REGIONS,
    DIRECTION_CLASSES,
    DIRECTION_TOLERANCE,
    EDGE_CATEGORIES,
    angular_error,
    cam_sector_profile,
    cam_target_layer,
    category_ratios,
    compute_cams,
    localization_ratio,
    pointing_hit,
    rank_correlation,
    sector_categories,
    wafer_regions,
)
from src.evaluation.run_context import download_checkpoint, load_clean_arrays, load_run_context
from src.models.factory import build_model_from_config, load_checkpoint
from src.training.tracking import committed_version, git_tags, resolve_tracking_uri
from src.training.trainer import predict, resolve_device

DEFAULT_OUTPUT = "reports/final/gradcam.md"
CAM_BATCH = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM analysis of a trained model.")
    parser.add_argument("--run-id", required=True, help="MLflow run id of the model, e.g. the deliverable.")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Split to analyze (default: validation).")
    parser.add_argument("--examples", type=int, default=4, help="Examples per class in the figures.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for drawing examples and for the random-weight model.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Markdown report; figures and a JSON file are written next to it.")
    return parser.parse_args()


def run_thresholds(run_id: str, names: list[str], download_dir: str) -> tuple[np.ndarray, str]:
    """The run's tuned thresholds from its test evaluation if it had one, else 0.5 for every class."""
    if "test/test_evaluation.json" not in {a.path for a in MlflowClient().list_artifacts(run_id, "test")}:
        return np.full(len(names), 0.5), "0.5 for every class (the run has no tuned thresholds)"
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="test/test_evaluation.json", dst_path=download_dir)
    thresholds = json.loads(Path(path).read_text(encoding="utf-8"))["thresholds"]
    return np.array([thresholds[name] for name in names]), "the run's tuned thresholds, as delivered"


def cams(model: nn.Module, layer: nn.Module, inputs: torch.Tensor, classes: list[int], device: torch.device) -> np.ndarray:
    """compute_cams in batches, so that long lists of samples fit in memory."""
    parts = [compute_cams(model, layer, inputs[i : i + CAM_BATCH].to(device), classes[i : i + CAM_BATCH]) for i in range(0, len(classes), CAM_BATCH)]
    return np.concatenate(parts) if parts else np.empty((0, *inputs.shape[-2:]))


def overlay(ax: plt.Axes, image: np.ndarray, cam: np.ndarray, title: str) -> None:
    """The wafer map in grays (white outside, gray pass, black fail) with the heatmap on top."""
    ax.imshow(image, cmap="gray_r", vmin=0, vmax=2)
    ax.imshow(np.ma.masked_where(image == 0, cam), cmap="jet", alpha=0.45, vmin=0, vmax=1)
    # An all-zero map renders as uniform blue, which could pass for weak evidence everywhere
    ax.set_title(title + ("\n(empty map: no positive evidence)" if cam.max() <= 0 else ""), fontsize=8)
    ax.axis("off")


def fmt(value: float, spec: str) -> str:
    return "–" if math.isnan(value) else format(value, spec)


def nanmean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return float(np.mean(finite)) if finite else math.nan


def edge_profile_diagnostic(positions: np.ndarray, trained: np.ndarray, randomized: np.ndarray, maps: np.ndarray) -> dict[str, Any]:
    """
    Where along the edge the heatmaps of a partial-edge class put their mass: on its failed sectors,
    where failed and intact parts meet, or on the intact part. Only samples with both failed and
    intact sectors are usable; a full ring or an edge without failed sectors has no such contrast.
    """
    samples, coverages = [], []
    ratios: dict[str, dict[str, list[float]]] = {"trained": {k: [] for k in EDGE_CATEGORIES}, "random": {k: [] for k in EDGE_CATEGORIES}}
    preferred: dict[str, list[str]] = {"trained": [], "random": []}
    sector_shares: dict[str, list[float]] = {k: [] for k in EDGE_CATEGORIES}
    for k, p in enumerate(positions):
        result = edge_sector_failures(maps[p])
        if result is None or result[0].all() or not result[0].any():
            continue
        categories = sector_categories(result[0])
        coverages.append(float(result[0].mean()))
        for kind in EDGE_CATEGORIES:
            sector_shares[kind].append(float((categories == kind).mean()))
        for model_kind, heatmap in (("trained", trained[k]), ("random", randomized[k])):
            profile = cam_sector_profile(heatmap, maps[p])
            if profile is None:
                continue
            sample_ratios = category_ratios(profile, categories)
            for kind in EDGE_CATEGORIES:
                ratios[model_kind][kind].append(sample_ratios[kind])
            present = {kind: value for kind, value in sample_ratios.items() if not math.isnan(value)}
            preferred[model_kind].append(max(present, key=present.get))
        samples.append(int(k))
    summary: dict[str, Any] = {
        "usable_samples": len(samples),
        "sample_positions": samples,
        "median_edge_coverage": float(np.median(coverages)) if coverages else math.nan,
        "sector_share": {kind: nanmean(values) for kind, values in sector_shares.items()},
    }
    for model_kind in ("trained", "random"):
        summary[model_kind] = {
            "mean_ratio": {kind: nanmean(values) for kind, values in ratios[model_kind].items()},
            "preferred_share": {kind: float(np.mean([p == kind for p in preferred[model_kind]])) if preferred[model_kind] else math.nan for kind in EDGE_CATEGORIES},
            "samples_with_heatmap": len(preferred[model_kind]),
        }
    return summary


def plot_edge_profiles(path: Path, chosen: list[int], positions: np.ndarray, heatmaps: np.ndarray, maps: np.ndarray, indices: np.ndarray, name: str) -> None:
    """Per sample: the map with its failed edge sectors in red, the heatmap, and both along the angle."""
    figure, axes = plt.subplots(len(chosen), 3, figsize=(9.6, 2.4 * len(chosen)), squeeze=False)
    centers = np.arange(36) * 10 - 175
    for row, k in enumerate(chosen):
        image = maps[positions[k]]
        failed, _ = edge_sector_failures(image)
        wafer = image != 0
        band = edge_band(wafer, 2)
        ys, xs = np.nonzero(band)
        sectors = angular_sectors(ys, xs, tuple(np.argwhere(wafer).mean(axis=0)))
        marked = np.zeros(image.shape, dtype=bool)
        marked[ys[failed[sectors]], xs[failed[sectors]]] = True
        axes[row][0].imshow(image, cmap="gray_r", vmin=0, vmax=2)
        axes[row][0].imshow(np.ma.masked_where(~marked, marked), cmap="autumn", alpha=0.8)
        axes[row][0].set_title(f"{name} #{indices[positions[k]]}: failed edge sectors in red", fontsize=8)
        axes[row][0].axis("off")
        overlay(axes[row][1], image, heatmaps[k], "Grad-CAM")
        profile = cam_sector_profile(heatmaps[k], image)
        ax = axes[row][2]
        for center, is_failed in zip(centers, failed, strict=True):
            if is_failed:
                ax.axvspan(center - 5, center + 5, color="tab:red", alpha=0.25, linewidth=0)
        if profile is not None:
            ax.plot(centers, profile, color="tab:blue")
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlabel("angle from center (0 = right, 90 = down)", fontsize=7)
        ax.set_ylabel("heatmap mass share", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_title("red: failed sectors; blue: heatmap", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    code_version = committed_version(git_tags())
    mlflow.set_tracking_uri(resolve_tracking_uri())
    device = resolve_device()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ctx = load_run_context(args.run_id, tmp_dir)
        model, checkpoint = load_checkpoint(download_checkpoint(ctx, tmp_dir), device)
        names = checkpoint["class_names"]
        thresholds, thresholds_source = run_thresholds(args.run_id, names, tmp_dir)

    # The same architecture with random weights, for the randomization check
    model_cfg = ModelConfig.model_validate({**checkpoint["model_config"], "pretrained": False})
    torch.manual_seed(args.seed)
    random_model = build_model_from_config(model_cfg, checkpoint["num_classes"], checkpoint["in_channels"]).to(device).eval()
    layer, random_layer = cam_target_layer(model, model_cfg.name), cam_target_layer(random_model, model_cfg.name)

    images, labels = load_clean_arrays(ctx)
    split = load_split(ctx.split_cfg.output_dir)
    indices = split.val_idx if args.split == "val" else split.test_idx
    dataset = DiscreteMapDataset(images, labels, indices, ctx.num_values, checkpoint["encoding"])
    y_true, y_prob, _ = predict(model, DataLoader(dataset, batch_size=256), nn.BCEWithLogitsLoss(), device)
    y_true, y_pred = y_true.astype(int), (y_prob >= thresholds).astype(int)
    inputs = torch.stack([dataset[i][0] for i in range(len(dataset))])
    maps = images[indices]
    single_correct = (y_true.sum(axis=1) == 1) & (y_true == y_pred).all(axis=1)
    rng = np.random.default_rng(args.seed)

    # 1. Localization, on every eligible sample
    localization: list[dict[str, Any]] = []
    for name, region_name in CLASS_REGIONS.items():
        if name not in names:
            continue
        c = names.index(name)
        positions = np.flatnonzero(single_correct & (y_true[:, c] == 1))
        if positions.size == 0:
            continue
        trained = cams(model, layer, inputs[positions], [c] * positions.size, device)
        randomized = cams(random_model, random_layer, inputs[positions], [c] * positions.size, device)
        ratios, random_ratios, hits, random_hits, correlations, area_shares = [], [], [], [], [], []
        for k, p in enumerate(positions):
            regions = wafer_regions(maps[p])
            region, wafer = regions[region_name], regions["wafer"]
            ratios.append(localization_ratio(trained[k], region, wafer))
            random_ratios.append(localization_ratio(randomized[k], region, wafer))
            hits.append(pointing_hit(trained[k], region))
            random_hits.append(pointing_hit(randomized[k], region))
            correlations.append(rank_correlation(trained[k], randomized[k], wafer))
            area_shares.append(region.sum() / wafer.sum())
        localization.append(
            {
                "class": name,
                "region": region_name,
                "samples": int(positions.size),
                # Chance level of the pointing game: a point drawn at random on the wafer
                "region_share_of_wafer": float(np.mean(area_shares)),
                "ratio_trained": nanmean(ratios),
                "ratio_random": nanmean(random_ratios),
                "hits_trained": nanmean(hits),
                "hits_random": nanmean(random_hits),
                "correlation_trained_random": nanmean(correlations),
                "empty_maps_trained": int(sum(math.isnan(r) for r in ratios)),
                "empty_maps_random": int(sum(math.isnan(r) for r in random_ratios)),
            }
        )

    # 1b. Direction, for defects on part of the edge, and where along the edge the heatmaps sit
    direction: list[dict[str, Any]] = []
    edge_profiles: list[dict[str, Any]] = []
    for name in DIRECTION_CLASSES:
        if name not in names:
            continue
        c = names.index(name)
        positions = np.flatnonzero(single_correct & (y_true[:, c] == 1))
        if positions.size == 0:
            continue
        trained = cams(model, layer, inputs[positions], [c] * positions.size, device)
        randomized = cams(random_model, random_layer, inputs[positions], [c] * positions.size, device)
        errors = [angular_error(trained[k], maps[p]) for k, p in enumerate(positions)]
        random_errors = [angular_error(randomized[k], maps[p]) for k, p in enumerate(positions)]
        along_edge = edge_profile_diagnostic(positions, trained, randomized, maps)
        along_edge["class"] = name
        edge_profiles.append(along_edge)

        def within(values: list[float]) -> float:
            finite = [v for v in values if not math.isnan(v)]
            return float(np.mean([v <= DIRECTION_TOLERANCE for v in finite])) if finite else math.nan

        direction.append(
            {
                "class": name,
                "samples": int(positions.size),
                "median_error_trained": float(np.nanmedian(errors)) if not all(math.isnan(e) for e in errors) else math.nan,
                "median_error_random": float(np.nanmedian(random_errors)) if not all(math.isnan(e) for e in random_errors) else math.nan,
                "within_tolerance_trained": within(errors),
                "within_tolerance_random": within(random_errors),
            }
        )

    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    examples_path = output.with_name(output.stem + "_examples.png")
    confusions_path = output.with_name(output.stem + "_edge_confusions.png")
    profiles_path = output.with_name(output.stem + "_edge_profiles.png")
    for stale in (examples_path, confusions_path, profiles_path):  # a figure from an earlier run must not pass for this one
        stale.unlink(missing_ok=True)

    # 1c. Edge profiles of a few random usable samples
    for row in edge_profiles:
        c = names.index(row["class"])
        positions = np.flatnonzero(single_correct & (y_true[:, c] == 1))
        usable = row["sample_positions"]
        if usable:
            # A generator of its own, so that adding this figure leaves the examples drawn for the other figures unchanged
            profile_rng = np.random.default_rng([args.seed, 1])
            chosen = sorted(profile_rng.choice(usable, size=min(args.examples + 2, len(usable)), replace=False).tolist())
            heatmaps = cams(model, layer, inputs[positions[chosen]], [c] * len(chosen), device)
            plot_edge_profiles(profiles_path, list(range(len(chosen))), positions[chosen], heatmaps, maps, indices, row["class"])

    # 2. Examples for every class
    figure, axes = plt.subplots(len(names), args.examples, figsize=(2.0 * args.examples, 2.1 * len(names)), squeeze=False)
    for c, name in enumerate(names):
        positions = np.flatnonzero(single_correct & (y_true[:, c] == 1))
        chosen = np.sort(rng.choice(positions, size=min(args.examples, positions.size), replace=False)) if positions.size else np.array([], dtype=int)
        heatmaps = cams(model, layer, inputs[chosen], [c] * chosen.size, device)
        for k in range(args.examples):
            if k < chosen.size:
                overlay(axes[c][k], maps[chosen[k]], heatmaps[k], f"{name} #{indices[chosen[k]]}")
            else:
                axes[c][k].axis("off")
    figure.suptitle("Grad-CAM for the labeled class, correctly classified single-defect samples drawn at random", fontsize=10)
    figure.tight_layout()
    figure.savefig(examples_path, dpi=120)
    plt.close(figure)

    # 3. Edge_Loc predicted as Edge_Ring
    confusions: list[dict[str, Any]] = []
    if {"Edge_Loc", "Edge_Ring"} <= set(names):
        el, er = names.index("Edge_Loc"), names.index("Edge_Ring")
        swapped = np.flatnonzero((y_true[:, el] == 1) & (y_true[:, er] == 0) & (y_pred[:, er] == 1) & (y_pred[:, el] == 0))
        if swapped.size:
            ring_maps = cams(model, layer, inputs[swapped], [er] * swapped.size, device)
            loc_maps = cams(model, layer, inputs[swapped], [el] * swapped.size, device)
            figure, axes = plt.subplots(swapped.size, 3, figsize=(6.3, 2.2 * swapped.size), squeeze=False)
            for k, p in enumerate(swapped):
                coverage = edge_coverage(maps[p])
                confusions.append({"index": int(indices[p]), "edge_coverage": coverage, "p_edge_loc": float(y_prob[p, el]), "p_edge_ring": float(y_prob[p, er])})
                axes[k][0].imshow(maps[p], cmap="gray_r", vmin=0, vmax=2)
                axes[k][0].set_title(f"#{indices[p]}, edge coverage {coverage:.2f}", fontsize=8)
                axes[k][0].axis("off")
                overlay(axes[k][1], maps[p], ring_maps[k], f"Edge_Ring (predicted), p={y_prob[p, er]:.2f}")
                overlay(axes[k][2], maps[p], loc_maps[k], f"Edge_Loc (label), p={y_prob[p, el]:.2f}")
            figure.tight_layout()
            figure.savefig(confusions_path, dpi=120)
            plt.close(figure)

    split_name = "validation" if args.split == "val" else "test"
    lines = [
        "# Grad-CAM analysis",
        "",
        f"Model: MLflow run {args.run_id} ({model_cfg.name}); {split_name} split; decisions at {thresholds_source}. "
        f"Heatmaps from the last convolutional block (7x7, upsampled to the input size). Code: commit `{code_version}`.",
        "",
        "## Localization",
        "",
        "For classes that lie in a known zone of the wafer (center, ring or edge, concentric zones that partition it), on every",
        "correctly classified single-defect sample. **Mass ratio**: the share of heatmap mass in the zone divided by the zone's",
        "share of the wafer (1 = no focus). **Pointing game**: how often the heatmap's peak falls in the zone; a point drawn at",
        "random on the wafer would hit it as often as the zone's share of the wafer. The control is the same architecture with",
        "random weights: its maps should not focus, nor resemble the trained model's (rank correlation near 0).",
        "",
        "| Class | Zone | Samples | Zone share of wafer (chance) | Peak in zone, trained | Peak in zone, random weights "
        "| Mass ratio, trained | Mass ratio, random weights | Correlation with random |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in localization:
        lines.append(
            f"| {row['class']} | {row['region']} | {row['samples']} | {row['region_share_of_wafer']:.0%} | {fmt(row['hits_trained'], '.0%')} | "
            f"{fmt(row['hits_random'], '.0%')} | {fmt(row['ratio_trained'], '.2f')} | {fmt(row['ratio_random'], '.2f')} | {fmt(row['correlation_trained_random'], '.2f')} |"
        )
    empty = [r for r in localization if r["empty_maps_trained"] or r["empty_maps_random"]]
    if empty:
        lines += [
            "",
            "Empty heatmaps (no positive evidence at the layer, so no peak and no mass) are left out of these figures; "
            + "; ".join(f"{r['class']}: {r['empty_maps_trained']} trained, {r['empty_maps_random']} random" for r in empty)
            + ". A dash means every map was empty.",
        ]
    if direction:
        tolerance = f"{DIRECTION_TOLERANCE:.0f}"
        lines += [
            "",
            "## Direction along the edge",
            "",
            "For defects on part of the edge: the angle between where the heatmap's mass points and where the edge's failed dies lie,",
            f"seen from the wafer's center. A random direction falls within {tolerance} degrees 25% of the time. Direction is measured",
            "because a 7x7 heatmap resolves it well but radius only to about one cell, its peak sitting one cell inward of an edge defect.",
            "",
            f"| Class | Samples | Median angle, trained | Within {tolerance} degrees, trained | Median angle, random weights | Within {tolerance} degrees, random weights |",
            "|---|---|---|---|---|---|",
        ]
        for row in direction:
            lines.append(
                f"| {row['class']} | {row['samples']} | {fmt(row['median_error_trained'], '.0f')} | {fmt(row['within_tolerance_trained'], '.0%')} | "
                f"{fmt(row['median_error_random'], '.0f')} | {fmt(row['within_tolerance_random'], '.0%')} |"
            )
    for row in edge_profiles:
        shares = row["sector_share"]
        lines += [
            "",
            f"## Where along the edge the {row['class']} heatmaps fall",
            "",
            "The 36 edge sectors of each map are failed (most of their dies failed, as in the edge coverage) or intact; a sector",
            "next to one of the other kind is a boundary, where failed and intact parts of the edge meet. For each kind: its share",
            "of the heatmap mass (beyond 0.4 of the radius) divided by its share of the sectors, so 1 means no preference. A",
            "heatmap pointing at the defect favors failed sectors; at the ends of the failed arc, boundaries; at the part of the",
            f"edge that did not fail, intact sectors. {row['usable_samples']} usable samples (both failed and intact sectors; median",
            f"edge coverage {row['median_edge_coverage']:.2f}); on average {shares['defect']:.0%} of sectors are failed, {shares['boundary']:.0%}",
            f"boundaries and {shares['intact']:.0%} intact.",
            "",
            "| Model | Samples with a heatmap | Ratio, failed | Ratio, boundary | Ratio, intact | Most favored: failed | boundary | intact |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for model_kind, label_text in (("trained", "Trained"), ("random", "Random weights")):
            stats = row[model_kind]
            mean, favored = stats["mean_ratio"], stats["preferred_share"]
            lines.append(
                f"| {label_text} | {stats['samples_with_heatmap']} | {fmt(mean['defect'], '.2f')} | {fmt(mean['boundary'], '.2f')} | {fmt(mean['intact'], '.2f')} | "
                f"{fmt(favored['defect'], '.0%')} | {fmt(favored['boundary'], '.0%')} | {fmt(favored['intact'], '.0%')} |"
            )
        if profiles_path.exists():
            lines += ["", f"![Edge profiles]({profiles_path.name})"]
    lines += ["", "## Examples", "", f"![Grad-CAM examples]({examples_path.name})", ""]
    if confusions:
        lines += [
            "## Edge_Loc predicted as Edge_Ring",
            "",
            f"All {len(confusions)} such samples of the {split_name} split, with the heatmaps of the predicted and the labeled class",
            "and the edge coverage of each map (share of the circumference with mostly failed dies; see docs/methodology.md).",
            "",
            f"![Edge confusions]({confusions_path.name})",
            "",
        ]
    output.write_text("\n".join(lines), encoding="utf-8")
    details = {
        "run_id": args.run_id,
        "split": args.split,
        "git_commit": code_version,
        "thresholds": dict(zip(names, thresholds.tolist(), strict=True)),
        "localization": localization,
        "direction": direction,
        "edge_confusions": confusions,
    }
    output.with_suffix(".json").write_text(json.dumps(details, indent=2), encoding="utf-8")

    with mlflow.start_run(run_id=args.run_id):
        mlflow.log_dict(details, "explain/gradcam.json")
        for path in (examples_path, confusions_path, profiles_path):
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="explain")
    print("\n".join(lines))
    print(f"Saved: {Path(args.output).as_posix()} with its figures and JSON")


if __name__ == "__main__":
    main()
