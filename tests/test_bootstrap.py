import numpy as np
import pytest

from src.evaluation.bootstrap import bootstrap_ci, paired_difference, resample_weights, sample_errors


def test_sample_errors_counts_any_wrong_label():
    y_true = np.array([[1, 0], [0, 1], [1, 1]])
    y_pred = np.array([[1, 0], [1, 1], [1, 0]])
    assert sample_errors(y_true, y_pred).tolist() == [False, True, True]


def test_resample_weights_draw_n_samples_each_time():
    weights = resample_weights(n_samples=50, n_resamples=20, seed=0)
    assert weights.shape == (20, 50) and np.all(weights.sum(axis=1) == 50)
    assert np.array_equal(weights, resample_weights(50, 20, seed=0))  # reproducible


def test_bootstrap_interval_contains_the_seed_averaged_errors():
    rng = np.random.default_rng(1)
    errors = rng.random((3, 400)) < 0.05  # 3 seeds, about 20 errors each
    result = bootstrap_ci(errors, resample_weights(400, 2000, seed=0))
    assert result["errors"] == pytest.approx(errors.sum(axis=1).mean())
    assert result["ci_low"] < result["errors"] < result["ci_high"]


def test_paired_difference_of_identical_models_is_exactly_zero():
    errors = np.random.default_rng(2).random((3, 300)) < 0.1
    result = paired_difference(errors, errors, resample_weights(300, 500, seed=0))
    assert result["difference"] == result["ci_low"] == result["ci_high"] == 0.0


def test_pairing_shrinks_the_interval_when_models_share_hard_samples():
    rng = np.random.default_rng(3)
    hard = rng.random(1000) < 0.08  # samples both models get wrong
    a = np.vstack([hard] * 3)
    b = np.vstack([hard | (rng.random(1000) < 0.01) for _ in range(3)])  # a few extra errors
    weights = resample_weights(1000, 2000, seed=0)
    paired = paired_difference(a, b, weights)
    # An unpaired interval would add both models' sampling variances; the paired one cancels the shared errors
    unpaired_width = (bootstrap_ci(a, weights)["ci_high"] - bootstrap_ci(a, weights)["ci_low"]) + (bootstrap_ci(b, weights)["ci_high"] - bootstrap_ci(b, weights)["ci_low"])
    assert paired["ci_high"] - paired["ci_low"] < unpaired_width / 3
    assert paired["difference"] < 0 and paired["share_a_fewer"] > 0.9
