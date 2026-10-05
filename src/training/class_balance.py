# Numpy-only helpers for handling class imbalance in multi-label training.

from typing import Literal

import numpy as np

PosWeightMode = Literal["none", "raw", "sqrt"]


def compute_pos_weight(labels: np.ndarray, mode: PosWeightMode = "sqrt", max_value: float | None = None) -> np.ndarray | None:
    """
    Per-class positive weight for BCEWithLogitsLoss, base on negative / positive.

    Must be computed from the TRAINING split only: using val/test labels would leak their class balance into training.

    - "none": no weighting (return None, i.e. plain BCE).
    - "raw": negatives / positives as is. For every rare classes this can exceed 100 and make the gradients of early training unstable.
    - "sqrt": square root of the raw ratio, a common way to damp extreme weights.
    `max_value` caps the weight after damping.
    """
    if mode == "none":
        return None
    if labels.ndim != 2:
        raise ValueError("pos_weight only applies to multi-label (2D) labels")

    positives = labels.sum(axis=0).astype(np.float64)
    if np.any(positives == 0):
        missing = np.where(positives == 0)[0].tolist()
        raise ValueError(f"Classes with no positive sample in the training labels: {missing}")
    negatives = labels.shape[0] - positives
    ratio = negatives / positives

    if mode == "sqrt":
        ratio = np.sqrt(ratio)
    elif mode != "raw":
        raise ValueError(f"Unsupported pos_weight mode: {mode}")

    if max_value is not None:
        ratio = np.minimum(ratio, max_value)
    return ratio.astype(np.float32)


def expected_initial_bce(labels: np.ndarray, pos_weight: np.ndarray | None = None) -> float:
    """ """
    weights = np.ones(labels.shape, dtype=np.float64)
    if pos_weight is not None:
        weights = np.where(labels == 1, pos_weight[np.newaxis, :], 1.0)
    return float(np.log(2) * weights.mean())
