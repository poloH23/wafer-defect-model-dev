"""
Analysis the validation error of a trained run and attatch the results to that same MLflow run
(artifacts under analysis/), so each model and the analysis of its mistakes stay together.

Usage:
    python scripts/analysis_errors.py --run-id <MLflow run id>

Only the validation split is analyzed. The test split stays untouched until the final evaluation:
studying test errors while still developing the model would turn it into a second validation set.
"""

import argparse
import tempfile

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from src.common.config_schema import EdaConfig, load_named_config
from src.data.dataset import DiscreteMapDataset
from src.data.split import load_split
from src.eda.plotting import build_discrete_plot_kwargs
from src.evaluation.error_analysis import combination_confusions, error_confidence, error_records, label_error_counts, label_swaps, misclassified, plot_errors
from src.evaluation.run_context import load_clean_arrays, load_run_context
from src.models.factory import load_checkpoint
from src.training.trainer import predict, resolve_device

THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the validation errors of an MLflow run.")
    parser.add_argument("--run-id", required=True, help="MLflow run id of a training run.")
    parser.add_argument("--max-images", type=int, default=16, help="Maximum number of misclassified images to plot.")
    parser.add_argument("--plot_seed", type=int, default=0, help="Seed for choosing which misclassified images to plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device()
    with tempfile.TemporaryDirectory() as tmp_dir:
        ctx = load_run_context(args.run_id, tmp_dir)
        checkpoint_path = mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path="checkpoints/best_model.pt", dst_path=tmp_dir)
        model, checkpoint = load_checkpoint(checkpoint_path, device)

    images, labels = load_clean_arrays(ctx)
    val_idx = load_split(ctx.split_cfg.output_dir).val_idx
    val_set = DiscreteMapDataset(images, labels, val_idx, ctx.num_values, checkpoint["encoding"])
    loader = DataLoader(val_set, batch_size=ctx.train_cfg.batch_size, shuffle=False)

    # The loss value is not used here; predict() only needs a loss function to report one
    y_true, y_prob, _ = predict(model, loader, nn.BCEWithLogitsLoss(), device)
    y_true, y_pred = y_true.astype(int), (y_prob >= THRESHOLD).astype(int)
    names = ctx.dataset_cfg.class_names

    errors = misclassified(y_true, y_pred)
    swaps = label_swaps(y_true, y_pred, names)
    confusions = combination_confusions(y_true, y_pred, names)
    confidence = error_confidence(y_true, y_pred, y_prob)
    print(f"Misclassified validation samples: {errors.size} / {len(y_true)} (threshold: {THRESHOLD})")
    print("\nMost of frequent swaps (class missed -> class predicted instead):")
    for row in swaps[:10]:
        print(f"  {row['missed']:<12} -> {row['predicted_instead']:<12} {row['count']}")
    print("\nMost frequent combination confusion:")
    for row in confusions[:10]:
        print(f"  {row['true']:<35} -> {row['predicted']:<35} {row['count']}")
    print("\nConfidence of the wrong label decisions (uncertain = probability between 0.2 and 0.8):")
    for kind, stats in confidence.items():
        if stats["count"]:
            print(f"  {kind:<16} count {stats['count']:>3} | median probability {stats['median']:.3f} | uncertain {stats['share_uncertain']:.0%}")

    # Plot a random sample of errors: the first errors by index would all come from the same few combinations
    rng = np.random.default_rng(args.plot_seed)
    shown = np.sort(rng.choice(errors, size=min(args.max_images, errors.size), replace=False)) if errors.size else errors
    try:
        colors = load_named_config("eda", ctx.dataset_cfg.name, EdaConfig).plot.discrete_colors
    except FileNotFoundError:
        colors = None
    figure = plot_errors(images[val_idx[shown]], y_true[shown], y_pred[shown], names, sample_ids=val_idx[shown], plot_kwargs=build_discrete_plot_kwargs(colors) if colors else None)

    with mlflow.start_run(run_id=args.run_id):
        mlflow.log_dict = (
            {
                "threshold": THRESHOLD,
                "num_validation_samples": int(len(y_true)),
                "num_misclassified": int(errors.size),
                "label_errors": label_error_counts(y_true, y_pred, names),
                "label_swaps": swaps,
                "combination_confusions": confusions,
                "confidence": confidence,
                # Indices refer to the full dataset, e.g. for scripts/preview_augmentation.py --index
                "samples": error_records(errors, val_idx, y_true, y_pred, y_prob, names),
            },
            "analysis/val_errors.json",
        )
        mlflow.log_figure(figure, "analysis/val_errors.png")
    plt.close(figure)
    print(f"\nAnalysis attached to run {args.run_id} under analysis/")


if __name__ == "__main__":
    main()
