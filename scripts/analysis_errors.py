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
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from torch import nn
from torch.utils.data import DataLoader

from src.common.config_schema import EdaConfig, TrainConfig, load_config, load_dataset_config, load_named_config, load_split_config
from src.data.cleaning import apply_invalid_pixel_policy
from src.data.dataset import DiscreteMapDataset, load_npz_arrays
from src.data.split import load_split, load_split_meta
from src.eda.plotting import build_discrete_plot_kwargs
from src.evaluation.error_analysis import combination_confusions, label_error_counts, label_swaps, misclassified, plot_errors
from src.models.factory import load_checkpoint
from src.training.tracking import resolve_tracking_uri
from src.training.trainer import predict, resolve_device

THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the validation errors of an MLflow run.")
    parser.add_argument("--run-id", required=True, help="MLflow run id of a training run.")
    parser.add_argument("--max-images", type=int, default=16, help="Maximum number of misclassified images to plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(resolve_tracking_uri())
    run = MlflowClient().get_run(args.run_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # The config logged with the run, not the current YAML file, which may have changed since
        config_name = Path(run.data.tags["config_file"]).name
        config_path = mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path=f"config/{config_name}", dst_path=tmp_dir)
        checkpoint_path = mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path="checkpoints/best_model.pt", dst_path=tmp_dir)
        train_cfg = load_config(config_path, TrainConfig)
        device = resolve_device()
        model, checkpoint = load_checkpoint(checkpoint_path, device)

    split_cfg = load_split_config(train_cfg.split)
    dataset_cfg = load_dataset_config(split_cfg.dataset)
    # Lineage: analyze the run on exactly the split it was trained and validated with
    if load_split_meta(split_cfg.output_dir)["split_digest"] != run.data.params["split_digest"]:
        raise RuntimeError(f"Split '{train_cfg.split}' on disk differs from the one this run used.")

    images, labels = load_npz_arrays(dataset_cfg.source_path, dataset_cfg.image_key, dataset_cfg.label_key)

    if not dataset_cfg.num_pixel_values:
        raise ValueError("The number of pixel values is missing, dataset validation is unable to proceed.")
    images, _ = apply_invalid_pixel_policy(images, dataset_cfg.num_pixel_values, dataset_cfg.invalid_pixel_policy)
    val_idx = load_split(split_cfg.output_dir).val_idx
    val_set = DiscreteMapDataset(images, labels, val_idx, dataset_cfg.num_pixel_values, checkpoint["encoding"])
    loader = DataLoader(val_set, batch_size=train_cfg.batch_size, shuffle=False)

    # The loss value is not used here; predict() only needs a loss function to report one
    y_true, y_prob, _ = predict(model, loader, nn.BCEWithLogitsLoss(), device)
    y_true, y_pred = y_true.astype(int), (y_prob >= THRESHOLD).astype(int)
    names = dataset_cfg.class_names

    errors = misclassified(y_true, y_pred)
    swaps = label_swaps(y_true, y_pred, names)
    confusions = combination_confusions(y_true, y_pred, names)
    print(f"Misclassified validation samples: {errors.size} / {len(y_true)} (threshold: {THRESHOLD})")
    print("\nMost of frequent swaps (class missed -> class predicted instead):")
    for row in swaps[:10]:
        print(f"  {row['missed']:<12} -> {row['predicted_instead']:<12} {row['count']}")
    print("\nMost frequent combination confusion:")
    for row in confusions[:10]:
        print(f"  {row['true']:<35} -> {row['predicted']:<35} {row['count']}")

    #
    try:
        colors = load_named_config("eda", dataset_cfg.name, EdaConfig).plot.discrete_colors
    except FileNotFoundError:
        colors = None
    shown = errors[: args.max_images]
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
                "misclassified_dataset_indices": val_idx[errors].tolist(),
            },
            "analysis/val_errors.json",
        )
        mlflow.log_figure(figure, "analysis/val_errors.png")
    plt.close()
    print(f"\nAnalysis attached to run {args.run_id} under analysis/")


if __name__ == "__main__":
    main()
