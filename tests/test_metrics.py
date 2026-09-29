import numpy as np
import pytest

from src.training.metrics import multilabel_metrics


@pytest.fixture
def y_true() -> np.ndarray:
    return np.array([[1, 0], [0, 1], [1, 1], [0, 0]])


def test_perfect_predictions(y_true):
    m = multilabel_metrics(y_true, y_true.astype(float), class_names=["a", "b"])
    for key in ("macro_ap", "macro_f1", "micro_f1", "exact_match"):
        assert m[key] == pytest.approx(1.0)
    assert m["per_class"]["b"]["support"] == 2


def test_ap_is_threshold_free(y_true):
    # A monotonic rescaling changes every thresholded decision but not the ranking, so AP stays the same
    prob = np.array([[0.9, 0.2], [0.3, 0.8], [0.7, 0.6], [0.1, 0.4]])
    a = multilabel_metrics(y_true, prob)
    b = multilabel_metrics(y_true, prob**3)
    assert a["macro_ap"] == pytest.approx(b["macro_ap"])
    assert a["macro_f1"] != pytest.approx(b["macro_f1"])


def test_per_class_thresholds(y_true):
    prob = np.array([[0.9, 0.2], [0.3, 0.35], [0.7, 0.4], [0.1, 0.1]])
    assert multilabel_metrics(y_true, prob, thresholds=0.5)["per_class"]["class_1"]["recall"] == 0.0
    assert multilabel_metrics(y_true, prob, thresholds=np.array([0.5, 0.3]))["per_class"]["class_1"]["recall"] == 1.0


def test_exact_match_requires_every_label(y_true):
    prob = y_true.astype(float)
    prob[2, 1] = 0.0  # one wrong label in one sample
    assert multilabel_metrics(y_true, prob)["exact_match"] == pytest.approx(0.75)


def test_class_without_positives_is_excluded_from_macro_ap():
    y = np.array([[1, 0], [0, 0], [1, 0]])
    m = multilabel_metrics(y, y.astype(float))
    assert np.isnan(m["per_class"]["class_1"]["ap"])
    assert m["macro_ap"] == pytest.approx(1.0)


def test_shape_mismatch_raises(y_true):
    with pytest.raises(ValueError, match="shape"):
        multilabel_metrics(y_true, np.zeros((4, 3)))
