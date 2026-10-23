"""
Export a trained classifier as a self-contained ONNX model for deployment
(see src/inference/classifier.py for how it is used).

The graph takes raw wafer maps and returns class probabilities: the input encoding and the sigmoid
are part of it, so a deployment cannot get them subtly wrong, for instance by ordering the one-hot
channels differently. he decision thresholds are metadata rather than graph operations, so they can
be retuned without exporting the model again.
"""

import json
from pathlib import Path

import onnx
import torch
from torch import nn

from src.inference.classifier import FORMAT_VERSION, INPUT_NAME, OUTPUT_NAME


class InferenceWrapper(nn.Module):
    """Map of category ids (N, H, W) -> the training encoding -> the classifier -> probabilities (N, C)."""

    def __init__(self, model: nn.Module, num_values: int, encoding: str):
        super().__init__()
        if encoding not in ("onehot", "scaled"):
            raise ValueError(f"Unsupported encoding: {encoding}")
        self.model = model
        self.num_values = num_values
        self.encoding = encoding

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        maps = maps.to(torch.int64).unsqueeze(1)
        if self.encoding == "onehot":
            # Channel v is 1 where the pixel equals v, exactly as src.data.transforms.encode_discrete
            categories = torch.arange(self.num_values, device=maps.device).view(1, -1, 1, 1)
            inputs = (maps == categories).to(torch.float32)
        else:
            inputs = maps.to(torch.float32) / (self.num_values - 1)
        return torch.sigmoid(self.model(inputs))


def build_metadata(
    class_names: list[str], thresholds: list[float], num_pixel_values: int, image_shape: tuple[int, int], invalid_pixel_policy: str, provenance: dict[str, str]
) -> dict[str, str]:
    """ONNX metadata values are strings; structured values are stored as JSON."""
    if len(thresholds) != len(class_names):
        raise ValueError("One threshold per class is required.")
    return {
        "format_version": FORMAT_VERSION,
        "class_names": json.dumps(class_names),
        "thresholds": json.dumps([float(t) for t in thresholds]),
        "num_pixel_values": str(num_pixel_values),
        "image_shape": json.dumps(list(image_shape)),
        "invalid_pixel_policy": invalid_pixel_policy,
        "provenance": json.dumps(provenance),
    }


def export_classifier(model: nn.Module, num_values: int, encoding: str, image_shpae: tuple[int, int], path: str | Path, metadata: dict[str, str]) -> Path:
    """
    Export to a single ONNX file with a dynamic batch dimension and uint8 input, and write the metadata
    into it. The weights stay inside the file: the exporter's default would put them in a separate
    .onnx.data file, a format for models over the 2 GB protobuf limit.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = InferenceWrapper(model.cpu().eval(), num_values, encoding).eval()
    torch.onnx.export(
        wrapper,
        (torch.zeros((2, *image_shpae), dtype=torch.uint8),),
        str(path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        external_data=False,
        verbose=False,
    )
    onnx_model = onnx.load(str(path))
    for key, value in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(path))
    return path
