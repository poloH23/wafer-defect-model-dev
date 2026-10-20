"""
Per-class decision thresholds, tuned on the validation split only (docs/model_selection.md).
"""

import numpy as np
from sklearn.metrics import f1_score

# Candidate thresholds: 0.05, 0.10, ..., 0.95
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
DEFAULT_THRESHOLD = 0.5


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray, grid: np.ndarray = THRESHOLD_GRID) -> np.ndarray:
    """
    For each class, the threshold that maximizes that class's F1 on the given (validation) data.

    Ties go to the candidate closest to 0.5, the untuned default, so that a flat F1 curve does not
    move the threshold arbitrarily. A class without positive samples has F1 = 0 everywhere and keeps 0.5.
    """
    if y_true.shape != y_prob.shape or y_true.ndim != 2:
        raise ValueError(f"Expected y_true and y_prob with the same (N, C) shape, got {y_true.shape} and {y_prob.shape}")
    thresholds = np.empty(y_true.shape[1])
    for c in range(y_true.shape[1]):
        scores = np.array([f1_score(y_true[:, c], y_prob[:, c] >= t, zero_division=0) for t in grid])
        best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))
        thresholds[c] = grid[best[np.argmin(np.abs(grid[best] - DEFAULT_THRESHOLD))]]
    return thresholds
