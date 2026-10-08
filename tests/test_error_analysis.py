"""tests/test_error_analysis.py"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.evaluation.error_analysis import (  # noqa: E402
    combination_confusions,
    combination_name,
    label_error_counts,
    label_swaps,
    misclassified,
    plot_errors,
)

NAMES = ["Edge_Loc", "Edge_Ring", "Scratch"]


@pytest.fixture
def predictions() -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]])
    y_pred = np.array([[0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    return y_true, y_pred


def test_combination_name():
    assert combination_name(np.array([1, 0, 1]), NAMES) == "Edge_Loc+Scratch"
    assert combination_name(np.array([0, 0, 0]), NAMES) == "Normal"


def test_misclassified_positions(predictions):
    assert misclassified(*predictions).tolist() == [0, 1, 3]


def test_label_error_counts(predictions):
    counts = label_error_counts(*predictions, NAMES)
    assert counts["Edge_Loc"] == {"false_negatives": 2, "false_positives": 0}
    assert counts["Edge_Ring"] == {"false_negatives": 0, "false_positives": 2}
    assert counts["Scratch"] == {"false_negatives": 0, "false_positives": 1}


def test_label_swaps_pair_missed_and_added_classes(predictions):
    # Samples 0 and 1 both miss Edge_Loc and add Edge_Ring; sample 3 only adds a label, so no swap
    assert label_swaps(*predictions, NAMES) == [{"missed": "Edge_Loc", "predicted_instead": "Edge_Ring", "count": 2}]


def test_combination_confusions_are_sorted_by_count():
    y_true = np.array([[1, 0, 0]] * 3 + [[0, 0, 1]])
    y_pred = np.array([[0, 1, 0]] * 3 + [[0, 0, 0]])
    result = combination_confusions(y_true, y_pred, NAMES)
    assert result[0] == {"true": "Edge_Loc", "predicted": "Edge_Ring", "count": 3}
    assert result[1] == {"true": "Scratch", "predicted": "Normal", "count": 1}


def test_plot_errors_draws_one_panel_per_image(predictions):
    y_true, y_pred = predictions
    images = np.zeros((3, 8, 8), dtype=np.uint8)
    fig = plot_errors(images, y_true[:3], y_pred[:3], NAMES, sample_ids=np.array([10, 20, 30]))
    titled = [ax for ax in fig.axes if ax.get_title()]
    assert len(titled) == 3
    assert titled[0].get_title().startswith("#10")
    plt.close(fig)
