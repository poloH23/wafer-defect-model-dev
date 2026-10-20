"""
Grad-CAM heatmaps of a trained classifier, and two checks that they reflect what the model learned:

- Localization. Some defect classes lie in a known zone by definition: Center in the middle of the
  wafer, Edge_Loc and Edge_Ring on its edge, Donut on a ring in between. A heatmap focused on the
  defect puts more of its mass in that zone than the zone's share of the wafer area, and its peak
  falls in the zone more often than a point drawn at random on the wafer (the pointing game).
- Direction. Edge_Loc is a defect on part of the edge: its heatmap should point toward it, seen from
  the wafer's center. A 7x7 heatmap resolves direction well but radius only to about one cell, and its
  peak tends to sit one cell inward of an edge defect; the direction is therefore the reliable measure
  of where it points along the edge. This was established on synthetic wafers whose defect positions
  were known, before measuring the real model.
- Randomization (Adebayo et al., 2018, "Sanity Checks for Saliency Maps"). Heatmaps of the same
  architecture with random weights. Some saliency methods produce nearly the same maps for a trained
  and a random model: they reflect the input's structure, not the model's reasoning. Maps worth
  interpreting change when the weights are randomized, and lose their focus.
"""

import math

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy.stats import spearmanr
from torch import nn

# Three concentric zones that partition the wafer, as fractions of its radius: center up to 0.3,
# ring from 0.3 to 0.7, edge beyond. The edge zone is about 7 pixels wide on a 52x52 map, one cell of
# a 7x7 heatmap: a thinner zone would count as unfocused a heatmap that is on the defect but, at that
# resolution, necessarily spreads one cell inward.
CENTER_RADIUS = 0.3
EDGE_RADIUS = 0.7
# The zone each class lies in by definition; the other classes have no fixed position
CLASS_REGIONS = {"Center": "center", "Donut": "ring", "Edge_Loc": "edge", "Edge_Ring": "edge"}
# Classes whose defect has a direction from the wafer's center: a part of the edge
DIRECTION_CLASSES = ("Edge_Loc",)
# A random direction falls within this many degrees of the defect's direction 25% of the time
DIRECTION_TOLERANCE = 45.0


def cam_target_layer(model: nn.Module, name: str) -> nn.Module:
    """
    The last convolutional block of each supported architecture: the closest layer to the decision
    that is still spatial (7x7 for a 52x52 input with the small-input stem).
    """
    if name.startswith("resnet"):
        return model.layer4[-1]
    if name.startswith(("mobilenet", "efficientnet")):
        return model.features[-1]
    if name.startswith("shufflenet"):
        return model.conv5
    if name == "slim_resnet":
        return model.stages[-1]
    raise ValueError(f"No Grad-CAM layer defined for '{name}'")


def compute_cams(model: nn.Module, layer: nn.Module, inputs: torch.Tensor, classes: list[int]) -> np.ndarray:
    """One heatmap per input, for the class given for that input: (N, H, W), upsampled to the input size, each scaled to [0, 1]."""
    with GradCAM(model=model, target_layers=[layer]) as cam:
        return cam(input_tensor=inputs, targets=[ClassifierOutputTarget(c) for c in classes])


def wafer_regions(image: np.ndarray, empty_value: int = 0) -> dict[str, np.ndarray]:
    """Boolean masks of the wafer and of its three concentric zones (center, ring, edge), which partition it."""
    wafer = image != empty_value
    center_y, center_x = np.argwhere(wafer).mean(axis=0)
    radius = math.sqrt(wafer.sum() / math.pi)
    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
    distance = np.hypot(yy - center_y, xx - center_x) / radius
    return {
        "wafer": wafer,
        "center": wafer & (distance <= CENTER_RADIUS),
        "ring": wafer & (distance > CENTER_RADIUS) & (distance <= EDGE_RADIUS),
        "edge": wafer & (distance > EDGE_RADIUS),
    }


def localization_ratio(cam: np.ndarray, region: np.ndarray, wafer: np.ndarray) -> float:
    """
    Share of the heatmap mass inside the region, divided by the region's share of the wafer area.
    1 means no focus (mass spread like area); above 1, the heatmap concentrates on the region.
    NaN for an empty heatmap.
    """
    mass = cam[wafer].sum()
    if mass <= 0:
        return math.nan
    return float((cam[region].sum() / mass) / (region.sum() / wafer.sum()))


def pointing_hit(cam: np.ndarray, region: np.ndarray) -> float:
    """1.0 if the heatmap's peak lies in the region, 0.0 if not, NaN for an empty heatmap (no peak)."""
    if cam.max() <= 0:
        return math.nan
    return float(region.flat[int(np.argmax(cam))])


def _direction(weights: np.ndarray, mask: np.ndarray, center: tuple[float, float]) -> float | None:
    """Direction (radians) of the weighted mass of the masked pixels, seen from `center`; None without mass."""
    ys, xs = np.nonzero(mask)
    w = weights[ys, xs]
    if w.sum() <= 0:
        return None
    angles = np.arctan2(ys - center[0], xs - center[1])
    return math.atan2(float((w * np.sin(angles)).sum()), float((w * np.cos(angles)).sum()))


def angular_error(cam: np.ndarray, image: np.ndarray, fail_value: int = 2) -> float:
    """
    Angle in degrees (0 to 180) between where the heatmap's mass points and where the failed dies of
    the edge zone lie, both seen from the wafer's center. NaN for an empty heatmap or an edge without
    failed dies. Uniformly scattered background failures cancel out in the direction of the failed dies.
    """
    regions = wafer_regions(image)
    center = tuple(np.argwhere(regions["wafer"]).mean(axis=0))
    cam_direction = _direction(cam, regions["wafer"], center)
    defect_direction = _direction((image == fail_value).astype(float), regions["edge"], center)
    if cam_direction is None or defect_direction is None:
        return math.nan
    difference = (cam_direction - defect_direction + math.pi) % (2 * math.pi) - math.pi
    return abs(math.degrees(difference))


def rank_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Spearman correlation of two heatmaps over the masked pixels; NaN if either is constant there."""
    x, y = a[mask], b[mask]
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(spearmanr(x, y).statistic)
