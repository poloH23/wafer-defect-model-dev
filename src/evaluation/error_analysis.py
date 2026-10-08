# NumPy / matplotlib helpers to find and summarize the mistakes of a multi-label classifier

from collections import Counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def combination_name(row: np.ndarray, class_names: list[str]) -> str:
    """Readable name of a label vector, e.g. "Edge_Loc+Scratch"; an all-zero vector is "Normal"."""
    active = [name for name, flag in zip(class_names, row, strict=True) if flag]
    return "+".join(active) if active else "Normal"


def misclassified(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Positions of the samples whose predicted label set differs from the true one in any label."""
    return np.where((y_true != y_pred).any(axis=1))[0]


def label_error_counts(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, dict[str, int]]:
    """Per class: false negatives (defects missed) and false positives (false alarms)."""
    false_negatives = ((y_true == 1) & (y_pred == 0)).sum(axis=0)
    false_positives = ((y_true == 0) & (y_pred == 1)).sum(axis=0)
    return {name: {"false_negatives": int(false_negatives[c]), "false_positives": int(false_positives[c])} for c, name in enumerate(class_names)}


def label_swaps(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> list[dict[str, Any]]:
    """
    Within the same sample, a class that was missed while another class was wrongly predicted,
    e.g. Edge_Loc missed and Edge_Ring predicted instead. Counted over all samples, these pairs are
    the direct evidence that two classes are being confused with each other.
    """
    counts: Counter[tuple[str, str]] = Counter()
    for i in misclassified(y_true, y_pred):
        missed = np.where((y_true[i] == 1) & (y_pred[i] == 0))[0]
        added = np.where((y_true[i] == 0) & (y_pred[i] == 1))[0]
        for m in missed:
            for a in added:
                counts[(class_names[m], class_names[a])] += 1
    return [{"missed": m, "predicted_instead": a, "count": c} for (m, a), c in counts.most_common()]


def combination_confusions(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> list[dict[str, Any]]:
    """Most frequent (true combination -> predicted combination) mistakes, most common first."""
    counts = Counter((combination_name(y_true[i], class_names), combination_name(y_pred[i], class_names)) for i in misclassified(y_true, y_pred))
    return [{"true": t, "predicted": p, "count": c} for (t, p), c in counts.most_common()]


def plot_errors(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    sample_ids: np.ndarray | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    columns: int = 4,
) -> plt.Figure:  # type: ignore
    """Grid of misclassified images, each titled with its true and predicted label combinations."""
    rows = max(1, int(np.ceil(len(images) / columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4.4 * rows), squeeze=False)
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k >= len(images):
            continue
        ax.imshow(images[k], **(plot_kwargs or {}))
        header = f"#{sample_ids[k]}\n" if sample_ids is not None else ""
        ax.set_title(f"{header}true: {combination_name(y_true[k], class_names)}\npred: {combination_name(y_pred[k], class_names)}", fontsize=9)
    fig.tight_layout()
    return fig
