import numpy as np
import pytest

from src.training.class_balance import compute_pos_weight, expected_initial_bce

# ================= fixture: synthesised data =================


@pytest.fixture
def labels() -> np.ndarray:
    y = np.zeros((100, 2), dtype=int)
    y[:50, 0] = 1
    y[:4, 1] = 1
    return y


# ================= Test: compute_pos_weight =================


def test_none_returns_none(labels):
    assert compute_pos_weight(labels, mode="none") is None


def test_raw_ratio(labels):
    w = compute_pos_weight(labels, mode="raw")
    assert w.dtype == np.float32
    assert np.allclose(w, [1.0, 24.0])


def test_sqrt_damping(labels):
    assert np.allclose(compute_pos_weight(labels, mode="sqrt"), [1.0, np.sqrt(24.0)])


def test_cap_is_applied_after_damping(labels):
    assert np.allclose(compute_pos_weight(labels, mode="raw", max_value=10.0), [1.0, 10.0])


def test_class_without_positives_raises(labels):
    labels[:, 1] = 0
    with pytest.raises(ValueError, match="no positive"):
        compute_pos_weight(labels, mode="raw")


def test_single_label_input_raises():
    with pytest.raises(ValueError, match="multi-label"):
        compute_pos_weight(np.array([0, 1, 1]), mode="raw")


# ================= Test: expected_initial_bce =================


def test_expected_initial_bce_without_weights(labels):
    assert expected_initial_bce(labels) == pytest.approx(np.log(2))


def test_expected_initial_bce_with_weights():
    y = np.array([[1, 0], [0, 0]])
    # Three labels contribute ln2, the single positive contributes 3 * ln2
    assert expected_initial_bce(y, np.array([3.0, 1.0])) == pytest.approx(np.log(2) * (3 + 1 + 1 + 1) / 4)
