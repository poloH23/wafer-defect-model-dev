import numpy as np
import pytest

from src.evaluation.thresholds import tune_thresholds


def test_threshold_moves_to_where_the_classes_separate():
    y_true = np.array([[0], [0], [1], [1]])
    y_prob = np.array([[0.05], [0.12], [0.20], [0.30]])  # positives score low but above the negatives
    # 0.15 and 0.20 both separate them (a sample scoring exactly the threshold counts as positive);
    # the tie goes to 0.20, the closer to the 0.5 default
    assert tune_thresholds(y_true, y_prob)[0] == pytest.approx(0.20)


def test_ties_go_to_the_value_closest_to_one_half():
    # Perfectly separated with a wide gap: every threshold in (0.1, 0.9] gives F1 = 1
    y_true = np.array([[0], [1]])
    y_prob = np.array([[0.1], [0.95]])
    assert tune_thresholds(y_true, y_prob)[0] == pytest.approx(0.5)


def test_class_without_positives_keeps_the_default():
    assert tune_thresholds(np.zeros((3, 1), dtype=int), np.array([[0.2], [0.7], [0.9]]))[0] == pytest.approx(0.5)


def test_one_threshold_per_class():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=(50, 3))
    thresholds = tune_thresholds(y_true, rng.random((50, 3)))
    assert thresholds.shape == (3,) and np.all((thresholds >= 0.05) & (thresholds <= 0.95))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same"):
        tune_thresholds(np.zeros((3, 2)), np.zeros((3, 1)))
