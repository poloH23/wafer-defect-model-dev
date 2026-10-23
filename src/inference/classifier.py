"""
Reference inference for an exported wafer defect classifier, with NumPy and ONNX Runtime only.

The ONNX file is self-contained: its graph takes raw wafer maps (category ids) and returns class
probabilities, and its metadata holds the class names, the decision thresholds and the expected input.
This module is the executable specification of how to use it; a deployment can reimplement it in any
language from the same metadata.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.data.cleaning import apply_invalid_pixel_policy

INPUT_NAME = "wafer_map"
OUTPUT_NAME = "probabilities"
# Version of the metadata layout below, so that a consumer can refuse a file it does not understand
FORMAT_VERSION = "1"


@dataclass(frozen=True)
class ModelSpec:
    """What an exported model expects and how its outputs are read, from the ONNX metadata."""

    class_names: list[str]
    # Per-class decision thresholds on the probabilities, in the order of class_names
    thresholds: np.ndarray
    # Pixels are category ids from 0 to num_pixel_values - 1 (0 outside the wafer, 1 pass, 2 fail)
    num_pixel_values: int
    image_shape: tuple[int, int]
    # How out-of-range pixels are handled, the same policy as in training
    invalid_pixel_policy: str
    # Traceability: MLflow run, git commit, split digest, model name
    provenance: dict[str, str]

    @classmethod
    def from_metadata(cls, metadata: dict[str, str]) -> "ModelSpec":
        if metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported model metadata format {metadata.get('format_version')!r}, expected {FORMAT_VERSION!r}")
        class_names = json.loads(metadata["class_names"])
        thresholds = np.asarray(json.loads(metadata["thresholds"]), dtype=np.float32)
        if len(thresholds) != len(class_names):
            raise ValueError("The metadata has a different number of thresholds and class names")
        return cls(
            class_names=class_names,
            thresholds=thresholds,
            num_pixel_values=int(metadata["num_pixel_values"]),
            image_shape=tuple(json.loads(metadata["image_shape"])),
            invalid_pixel_policy=metadata["invalid_pixel_policy"],
            provenance=json.loads(metadata["provenance"]),
        )


class WaferDefectClassifier:
    """
    Multi-label wafer defect classifier from an exported ONNX file.

    Inputs are wafer maps of category ids, shaped (H, W) for one map or (N, H, W) for a batch.
    Out-of-range pixel values are handled with the training policy: imputed from their neighbors,
    or rejected, depending on the model's metadata.
    """

    def __init__(self, path: str | Path, threads: int | None = None, providers: tuple[str, ...] = ("CPUExecutionProvider",)):
        options = ort.SessionOptions()
        if threads is not None:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=list(providers))
        self.spec = ModelSpec.from_metadata(self.session.get_modelmeta().custom_metadata_map)

    def _prepare(self, maps: np.ndarray) -> np.ndarray:
        maps = np.asarray(maps)
        if maps.ndim == 2:
            maps = maps[np.newaxis]
        if maps.ndim != 3 or tuple(maps.shape[1:]) != self.spec.image_shape:
            raise ValueError(f"Expected maps shaped (N, {self.spec.image_shape[0]}, {self.spec.image_shape[1]}), got {maps.shape}")
        if not np.issubdtype(maps.dtype, np.integer):
            raise ValueError(f"Expected integer category ids, got {maps.dtype}")
        cleaned, _ = apply_invalid_pixel_policy(maps, self.spec.num_pixel_values, self.spec.invalid_pixel_policy)
        return cleaned.astype(np.uint8)

    def predict_proba(self, maps: np.ndarray) -> np.ndarray:
        """Class probabilities, (N, number of classes)."""
        return self.session.run([OUTPUT_NAME], {INPUT_NAME: self._prepare(maps)})[0]

    def predict(self, maps: np.ndarray) -> np.ndarray:
        """Boolean decisions, (N, number of classes), at the model's per-class thresholds."""
        return self.predict_proba(maps) >= self.spec.thresholds

    def predict_labels(self, maps: np.ndarray) -> list[list[str]]:
        """The names of the predicted defect classes of each map; an empty list for a map without defect."""
        return [[name for name, flag in zip(self.spec.class_names, row, strict=True) if flag] for row in self.predict(maps)]
