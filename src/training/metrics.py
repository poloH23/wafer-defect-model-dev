# NumPy / scikit-learn metrics for multi-label classification.
# Computed on the predictions of a whole epoch, so the function serves training-time validation, threshold tuning and the final test report.

import warnings
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_fscore_support


def multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: float | np.ndarray = 0.5,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """ """
    if y_true.shape != y_prob.shape or y_true.ndim != 2:
        raise ValueError(f"Excepted y_true and y_prob with the same (N, C) shape, got {y_true.shape} and {y_prob.shape}")

    num_classes = y_true.shape[1]
    names = class_names or [f"class_{i}" for i in range(num_classes)]
    if len(names) != num_classes:
        raise ValueError(f"Got {len(names)} class names for {num_classes} classes")

    y_true = y_true.astype(int)
    y_pred = (y_prob >= np.asarray(thresholds)).astype(int)

    # AP is undefined for a class with no positive sample; report NaN and leave it out of mean
    support = y_true.sum(axis=0)
    per_class_ap = np.full(num_classes, np.nan)
    for c in np.where(support > 0)[0]:
        per_class_ap[c] = average_precision_score(y_true[:, c], y_prob[:, c])

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        macro_ap = float(np.nanmean(per_class_ap))

    return {
        "macro_ap": macro_ap,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "exact_match": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            name: {
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
                "ap": float(per_class_ap[c]),
                "support": int(support[c]),
            }
            for c, name in enumerate(names)
        },
    }
