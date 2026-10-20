import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.evaluation.error_analysis import (  # noqa: E402
    combination_confusions,
    combination_name,
    error_confidence,
    error_records,
    label_error_counts,
    label_swaps,
    misclassified,
    plot_errors,
)

NAMES = ["Edge_Loc", "Edge_Ring", "Scratch"]

# ================= fixture: synthesised data =================


@pytest.fixture
def predictions() -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]])
    y_pred = np.array([[0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    return y_true, y_pred


# ================= Test: combination_name =================


def test_combination_name():
    assert combination_name(np.array([1, 0, 1]), NAMES) == "Edge_Loc+Scratch"
    assert combination_name(np.array([0, 0, 0]), NAMES) == "Normal"


# ================= Test: misclassified =================


def test_misclassified_positions(predictions):
    assert misclassified(*predictions).tolist() == [0, 1, 3]


# ================= Test: label_error_counts =================


def test_label_error_counts(predictions):
    counts = label_error_counts(*predictions, NAMES)
    assert counts["Edge_Loc"] == {"false_negatives": 2, "false_positives": 0}
    assert counts["Edge_Ring"] == {"false_negatives": 0, "false_positives": 2}
    assert counts["Scratch"] == {"false_negatives": 0, "false_positives": 1}


# ================= Test: label_swaps =================


def test_label_swaps_pair_missed_and_added_classes(predictions):
    # Samples 0 and 1 both miss Edge_Loc and add Edge_Ring; sample 3 only adds a label, so no swap
    assert label_swaps(*predictions, NAMES) == [{"missed": "Edge_Loc", "predicted_instead": "Edge_Ring", "count": 2}]


# ================= Test: combination_confusions =================


def test_combination_confusions_are_sorted_by_count():
    y_true = np.array([[1, 0, 0]] * 3 + [[0, 0, 1]])
    y_pred = np.array([[0, 1, 0]] * 3 + [[0, 0, 0]])
    result = combination_confusions(y_true, y_pred, NAMES)
    assert result[0] == {"true": "Edge_Loc", "predicted": "Edge_Ring", "count": 3}
    assert result[1] == {"true": "Scratch", "predicted": "Normal", "count": 1}


# ================= Test: plot_errors =================


def test_plot_errors_draws_one_panel_per_image(predictions):
    y_true, y_pred = predictions
    images = np.zeros((3, 8, 8), dtype=np.uint8)
    fig = plot_errors(images, y_true[:3], y_pred[:3], NAMES, sample_ids=np.array([10, 20, 30]))
    titled = [ax for ax in fig.axes if ax.get_title()]
    assert len(titled) == 3
    assert titled[0].get_title().startswith("#10")
    plt.close(fig)


# ================= Test: error_records =================


def test_error_records(predictions):
    y_true, y_pred = predictions
    y_prob = np.where(y_pred == 1, 0.9, 0.1)
    records = error_records(np.array([0]), np.array([100, 101, 102, 103, 104]), y_true, y_pred, y_prob, NAMES)
    assert records == [
        {
            "index": 100,
            "true": ["Edge_Loc", "Scratch"],
            "predicted": ["Edge_Ring", "Scratch"],
            "probabilities": {"Edge_Loc": 0.1, "Edge_Ring": 0.9, "Scratch": 0.9},
        }
    ]


# ================= Test: error_confidence =================


def test_error_confidence_separates_unsure_and_confident_mistakes():
    y_true = np.array([[1, 0], [1, 0], [0, 0]])
    y_prob = np.array([[0.45, 0.1], [0.02, 0.1], [0.1, 0.95]])
    y_pred = (y_prob >= 0.5).astype(int)
    result = error_confidence(y_true, y_pred, y_prob)
    assert result["false_negatives"] == {"count": 2, "median": pytest.approx(0.235), "share_uncertain": 0.5}
    assert result["false_positives"]["count"] == 1
    assert result["false_positives"]["share_uncertain"] == 0.0


def test_error_confidence_without_errors():
    y = np.array([[1, 0]])
    assert error_confidence(y, y, y.astype(float)) == {"false_negatives": {"count": 0}, "false_positives": {"count": 0}}
