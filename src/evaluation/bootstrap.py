"""
Uncertainty of test results that comes from the finite test set, by bootstrap resampling of test
samples. The other source of uncertainty, training randomness, is reported separately as the standard
deviation over seeds.

Resamples are represented by weights: how many times each test sample is drawn in each resample.
Counting errors on a resample is then a weighted sum, computed for every resample at once.
"""

import numpy as np


def sample_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per test sample, whether it is misclassified: any of its labels wrong."""
    return (np.asarray(y_true) != np.asarray(y_pred)).any(axis=1)


def resample_weights(n_samples: int, n_resamples: int, seed: int) -> np.ndarray:
    """(n_resamples, n_samples): how many times each sample is drawn in each resample of size n_samples."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_samples, np.full(n_samples, 1.0 / n_samples), size=n_resamples).astype(np.int32)


def resampled_mean_errors(errors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """errors: (seeds, n_samples) booleans. Returns, for each resample, the number of errors averaged over seeds."""
    return (np.atleast_2d(errors).astype(np.float64) @ weights.T.astype(np.float64)).mean(axis=0)


def _interval(values: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1.0 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)


def bootstrap_ci(errors: np.ndarray, weights: np.ndarray, level: float = 0.95) -> dict[str, float]:
    """Seed-averaged number of test errors, with a percentile bootstrap confidence interval."""
    low, high = _interval(resampled_mean_errors(errors, weights), level)
    return {"errors": float(np.atleast_2d(errors).sum(axis=1).mean()), "ci_low": low, "ci_high": high}


def paired_difference(errors_a: np.ndarray, errors_b: np.ndarray, weights: np.ndarray, level: float = 0.95) -> dict[str, float]:
    """
    Seed-averaged errors of A minus those of B (negative: A makes fewer errors), with a confidence
    interval from the same resamples for both, so that test samples hard for both cancel out.
    `share_a_fewer` is the share of resamples on which A makes fewer errors.
    """
    differences = resampled_mean_errors(errors_a, weights) - resampled_mean_errors(errors_b, weights)
    low, high = _interval(differences, level)
    point = float(np.atleast_2d(errors_a).sum(axis=1).mean() - np.atleast_2d(errors_b).sum(axis=1).mean())
    return {"difference": point, "ci_low": low, "ci_high": high, "share_a_fewer": float(np.mean(differences < 0))}
