"""
Export the delivered model as a self-contained ONNX file, verify it, register it, and write its
model card.

Usage:
    python scripts/export_onnx.py --run-id <MLflow run id of the delivered model>

1. Export: raw wafer maps in, class probabilities out; class names, thresholds and provenance in the
   file's metadata (src/models/export.py).
2. Verify, on the validation split: raw maps through the ONNX file and the reference inference
   (src/inference/classifier.py) against the PyTorch checkpoint on the training pipeline's inputs. Any
   decision that differs, or a probability difference above the tolerance, stops the script before
   anything is registered.
3. Register in the MLflow model registry, under an alias (default "champion"), so that a deployment
   loads models:/<name>@champion rather than a file path.
4. Write the model card from the test evaluation and benchmark reports.
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.onnx
import numpy as np
import onnx
import torch
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from torch import nn
from torch.utils.data import DataLoader

from src.common.config_schema import EvaluationConfig, load_config, resolve_path
from src.data.dataset import DiscreteMapDataset
from src.data.io import load_npz_arrays
from src.data.split import load_split
from src.evaluation.model_card import render_model_card
from src.evaluation.run_context import download_checkpoint, load_clean_arrays, load_run_context
from src.inference.classifier import WaferDefectClassifier
from src.models.export import build_metadata, export_classifier
from src.models.factory import load_checkpoint
from src.training.tracking import committed_version, git_tags, resolve_tracking_uri
from src.training.trainer import predict

DEFAULT_OUTPUT = "models/wafer_defect_classifier.onnx"
DEFAULT_CARD = "reports/final/model_card.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export, verify and register the delivered model.")
    parser.add_argument("--run-id", required=True, help="MLflow run id of the delivered model.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path of the ONNX file.")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="Largest accepted probability difference with PyTorch.")
    parser.add_argument("--register-name", default="wafer-defect-classifier", help="Name in the MLflow model registry.")
    parser.add_argument("--alias", default="champion", help="Alias given to the registered version.")
    parser.add_argument("--no-register", action="store_true", help="Export and verify only.")
    parser.add_argument("--evaluation-config", default="conf/evaluation/final_test.yaml", help="Test evaluation config, for the model card.")
    parser.add_argument("--test-report", default="reports/final/test_evaluation.json", help="Test evaluation details, for the model card.")
    parser.add_argument("--benchmark", default="reports/experiments/benchmark.json", help="Benchmark details, for the model card.")
    parser.add_argument("--card", default=DEFAULT_CARD, help="Path of the model card.")
    return parser.parse_args()


def delivered_thresholds(run_id: str, class_names: list[str], download_dir: str) -> list[float]:
    """The thresholds tuned for this model during the test evaluation, which the protocol delivers with it."""
    if "test/test_evaluation.json" not in {a.path for a in MlflowClient().list_artifacts(run_id, "test")}:
        raise FileNotFoundError(f"Run {run_id} has no tuned thresholds: export the model evaluated by scripts/evaluate_test.py")
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="test/test_evaluation.json", dst_path=download_dir)
    thresholds = json.loads(Path(path).read_text(encoding="utf-8"))["thresholds"]
    return [float(thresholds[name]) for name in class_names]


def verify(
    model: nn.Module,
    onnx_path: Path,
    raw_maps: np.ndarray,
    clean_maps: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    num_values: int,
    encoding: str,
    thresholds: list[float],
) -> dict[str, Any]:
    """The ONNX file through the reference inference on raw maps, against the PyTorch checkpoint on the training inputs."""
    loader = DataLoader(DiscreteMapDataset(clean_maps, labels, indices, num_values, encoding), batch_size=256, shuffle=False)
    _, expected, _ = predict(model, loader, nn.BCEWithLogitsLoss(), torch.device("cpu"))
    classifier = WaferDefectClassifier(onnx_path)
    actual = classifier.predict_proba(raw_maps[indices])
    batch = raw_maps[indices[:32]]
    one_by_one = np.concatenate([classifier.predict_proba(m) for m in batch])
    return {
        "samples": int(len(indices)),
        "max_abs_difference": float(np.abs(actual - expected).max()),
        "decision_mismatches": int(((actual >= np.array(thresholds)) != (expected >= np.array(thresholds))).any(axis=1).sum()),
        "batch_max_abs_difference": float(np.abs(one_by_one - classifier.predict_proba(batch)).max()),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    code_version = committed_version(git_tags())
    mlflow.set_tracking_uri(resolve_tracking_uri())

    with tempfile.TemporaryDirectory() as tmp_dir:
        ctx = load_run_context(args.run_id, tmp_dir)
        model, checkpoint = load_checkpoint(download_checkpoint(ctx, tmp_dir), torch.device("cpu"))
        names = checkpoint["class_names"]
        thresholds = delivered_thresholds(args.run_id, names, tmp_dir)

    raw_maps, labels = load_npz_arrays(ctx.dataset_cfg.source_path, ctx.dataset_cfg.image_key, ctx.dataset_cfg.label_key)
    clean_maps, _ = load_clean_arrays(ctx)
    image_shape = tuple(raw_maps.shape[1:])
    run = MlflowClient().get_run(args.run_id)
    provenance = {
        "model_name": ctx.train_cfg.model,
        "mlflow_run_id": args.run_id,
        "training_git_commit": run.data.tags.get("git_commit", "unknown"),
        "export_git_commit": code_version,
        "split_digest": run.data.params["split_digest"],
    }
    spec = {
        "class_names": names,
        "thresholds": thresholds,
        "num_pixel_values": ctx.num_values,
        "image_shape": image_shape,
        "invalid_pixel_policy": ctx.dataset_cfg.invalid_pixel_policy,
        "provenance": provenance,
    }

    # 1. Export
    output = resolve_path(args.output)
    metadata = build_metadata(names, thresholds, ctx.num_values, image_shape, ctx.dataset_cfg.invalid_pixel_policy, provenance)
    export_classifier(model, ctx.num_values, checkpoint["encoding"], image_shape, output, metadata)

    # 2. Verify, before anything is registered
    val_idx = load_split(ctx.split_cfg.output_dir).val_idx
    verification = verify(model, output, raw_maps, clean_maps, labels, val_idx, ctx.num_values, checkpoint["encoding"], thresholds)
    print(f"Verification on {verification['samples']} validation maps: {verification}")
    if verification["max_abs_difference"] > args.tolerance or verification["decision_mismatches"] or verification["batch_max_abs_difference"] > args.tolerance:
        raise RuntimeError("The ONNX export does not reproduce the PyTorch model; nothing was registered")
    digest = sha256(output)

    # 3. Log with the run, and register
    with mlflow.start_run(run_id=args.run_id):
        mlflow.log_artifact(str(output), artifact_path="export")
        mlflow.log_dict({"verification": verification, "sha256": digest, "metadata": metadata}, "export/export.json")
        if not args.no_register:
            example = raw_maps[val_idx[:4]]
            info = mlflow.onnx.log_model(
                onnx_model=onnx.load(str(output)),
                name="onnx_classifier",
                registered_model_name=args.register_name,
                signature=infer_signature(example, WaferDefectClassifier(output).predict_proba(example)),
                save_as_external_data=False,  # one self-contained file, as exported
                metadata={"sha256": digest, **provenance},
            )
            version = info.registered_model_version
            MlflowClient().set_registered_model_alias(args.register_name, args.alias, version)
            print(f"Registered {args.register_name} version {version} as @{args.alias}")

    # 4. Model card
    evaluation_cfg = load_config(args.evaluation_config, EvaluationConfig) if resolve_path(args.evaluation_config).exists() else None
    selected_label = None
    if evaluation_cfg is not None:
        selected = evaluation_cfg.selected
        selected_label = selected.experiment if selected.overrides == "none" else f"{selected.experiment} ({selected.overrides})"
    test_report_path, benchmark_path = resolve_path(args.test_report), resolve_path(args.benchmark)
    test_report = json.loads(test_report_path.read_text(encoding="utf-8")) if test_report_path.exists() else None
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.exists() else None
    benchmark_row = next((row for row in benchmark["results"] if row["model"] == provenance["model_name"]), None) if benchmark else None
    card = render_model_card(
        spec,
        verification,
        test_report,
        selected_label,
        evaluation_cfg.deliverable_seed if evaluation_cfg else 0,
        benchmark_row,
        digest,
    )
    card_path = resolve_path(args.card)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card, encoding="utf-8")
    print(f"Saved: {Path(args.output).as_posix()} (SHA-256 {digest[:12]}...) and {Path(args.card).as_posix()}")


if __name__ == "__main__":
    main()
