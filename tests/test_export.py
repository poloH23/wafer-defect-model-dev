from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from src.data.cleaning import apply_invalid_pixel_policy
from src.data.transforms import encode_discrete
from src.inference.classifier import WaferDefectClassifier
from src.models.export import InferenceWrapper, build_metadata, export_classifier
from src.models.factory import build_model

SHAPE = (52, 52)
CLASSES = [f"class_{i}" for i in range(8)]
THRESHOLDS = [0.5, 0.3, 0.7, 0.5, 0.45, 0.5, 0.55, 0.6]


def random_maps(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 3, size=(n, *SHAPE)).astype(np.uint8)


# ================= fixture: synthesised data =================


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[torch.nn.Module, Path]:
    torch.manual_seed(0)
    model = build_model("slim_resnet", num_classes=8, in_channels=3, width=8, depths=(1, 1, 1, 1)).eval()
    metadata = build_metadata(CLASSES, THRESHOLDS, 3, SHAPE, "neighbor_mode", {"mlflow_run_id": "abc123"})
    path = export_classifier(model, 3, "onehot", SHAPE, tmp_path_factory.mktemp("onnx") / "classifier.onnx", metadata)
    return model, path


class Capture(torch.nn.Module):
    """Stands in for the classifier and keeps what the wrapper feeds it."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.inputs = inputs
        return torch.zeros(inputs.shape[0], 8)


@pytest.mark.parametrize("encoding", ["onehot", "scaled"])
def test_wrapper_encodes_exactly_like_training(encoding):
    maps = random_maps(4)
    capture = Capture()
    InferenceWrapper(capture, num_values=3, encoding=encoding)(torch.from_numpy(maps))
    expected = np.stack([encode_discrete(m, 3, encoding) for m in maps])
    assert np.array_equal(capture.inputs.numpy(), expected)


def test_onnx_probabilities_and_decisions_match_pytorch(exported):
    model, path = exported
    maps = random_maps(64, seed=1)
    with torch.no_grad():
        expected = torch.sigmoid(model(torch.from_numpy(np.stack([encode_discrete(m, 3) for m in maps])))).numpy()
    classifier = WaferDefectClassifier(path)
    probabilities = classifier.predict_proba(maps)
    assert np.abs(probabilities - expected).max() < 1e-5
    assert np.array_equal(classifier.predict(maps), expected >= np.array(THRESHOLDS))


def test_batch_size_does_not_change_the_outputs(exported):
    classifier = WaferDefectClassifier(exported[1])
    maps = random_maps(8, seed=2)
    one_by_one = np.concatenate([classifier.predict_proba(m) for m in maps])  # each a single 2D map
    assert np.abs(one_by_one - classifier.predict_proba(maps)).max() < 1e-6


def test_the_file_is_self_contained(exported):
    path = exported[1]
    assert sorted(p.name for p in path.parent.iterdir()) == ["classifier.onnx"]
    spec = WaferDefectClassifier(path).spec
    assert spec.class_names == CLASSES and np.allclose(spec.thresholds, THRESHOLDS)
    assert (spec.num_pixel_values, spec.image_shape, spec.invalid_pixel_policy) == (3, SHAPE, "neighbor_mode")
    assert spec.provenance == {"mlflow_run_id": "abc123"}


def test_invalid_pixels_follow_the_training_policy(exported):
    classifier = WaferDefectClassifier(exported[1])
    maps = random_maps(2, seed=3)
    maps[0, 10, 10] = 3  # out of range, as in the raw dataset
    cleaned, report = apply_invalid_pixel_policy(maps, 3, "neighbor_mode")
    assert report["invalid_pixels"] == 1
    assert np.array_equal(classifier.predict_proba(maps), classifier.predict_proba(cleaned))


def test_labels_and_input_validation(exported):
    classifier = WaferDefectClassifier(exported[1])
    labels = classifier.predict_labels(random_maps(3, seed=4))
    assert len(labels) == 3 and all(set(row) <= set(CLASSES) for row in labels)
    with pytest.raises(ValueError, match="Expected maps shaped"):
        classifier.predict_proba(np.zeros((2, 40, 40), dtype=np.uint8))
    with pytest.raises(ValueError, match="integer"):
        classifier.predict_proba(np.zeros((1, *SHAPE), dtype=np.float32))


def test_unknown_metadata_format_is_refused(exported, tmp_path):
    onnx_model = onnx.load(str(exported[1]))
    for entry in onnx_model.metadata_props:
        if entry.key == "format_version":
            entry.value = "999"
    path = tmp_path / "future.onnx"
    onnx.save(onnx_model, str(path))
    with pytest.raises(ValueError, match="Unsupported model metadata format"):
        WaferDefectClassifier(path)
