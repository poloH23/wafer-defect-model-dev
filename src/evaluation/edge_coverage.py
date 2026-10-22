"""
A rule-based measure of how far a defect wraps around the wafer edge, used to check whether
Edge_Loc and Edge_Ring are separable in the data itself, independently of any trained model.

By definition, Edge_Ring is a defect along the whole edge and Edge_Loc a defect on part of it.
Edge coverage measures exactly that difference: the share of the wafer's circumference whose
outermost band consists mostly of failed dies.
"""

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score


def edge_band(wafer_mask: np.ndarray, width: int = 2) -> np.ndarray:
    """Wafer pixels within `width` pixels of the wafer boundary. Pixels on the image border count as boundary."""
    eroded = ndimage.binary_erosion(wafer_mask, iterations=width, border_value=0)
    return wafer_mask & ~eroded


def angular_sectors(ys: np.ndarray, xs: np.ndarray, center: tuple[float, float], num_sectors: int = 36) -> np.ndarray:
    """Sector index (0 to num_sectors - 1) of each pixel, by its angle around `center`, starting from the left (-180 degrees)."""
    angles = np.arctan2(ys - center[0], xs - center[1])
    return np.floor((angles + np.pi) / (2 * np.pi) * num_sectors).astype(int) % num_sectors


def edge_sector_failures(
    image: np.ndarray,
    empty_value: int = 0,
    fail_value: int = 2,
    band_width: int = 2,
    num_sectors: int = 36,
    sector_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    For each angular sector of the edge band, around the wafer centroid: whether at least
    `sector_threshold` of its dies failed, and whether it contains any band pixel at all.
    None for an image without a wafer.
    """
    wafer = image != empty_value
    band = edge_band(wafer, band_width)
    if not band.any():
        return None

    center = tuple(np.argwhere(wafer).mean(axis=0))
    ys, xs = np.nonzero(band)
    sectors = angular_sectors(ys, xs, center, num_sectors)
    totals = np.bincount(sectors, minlength=num_sectors)
    failed = np.bincount(sectors, weights=(image[ys, xs] == fail_value).astype(float), minlength=num_sectors)
    occupied = totals > 0
    mostly_failed = np.zeros(num_sectors, dtype=bool)
    mostly_failed[occupied] = failed[occupied] / totals[occupied] >= sector_threshold
    return mostly_failed, occupied


def edge_coverage(
    image: np.ndarray,
    empty_value: int = 0,
    fail_value: int = 2,
    band_width: int = 2,
    num_sectors: int = 36,
    sector_threshold: float = 0.5,
) -> float:
    """
    Split the edge band into `num_sectors` angular sectors around the wafer centroid, and return the
    share of sectors in which at least `sector_threshold` of the dies failed.

    About 1.0 means failures all around the edge (a ring), a small value means failures on part of
    the edge, and 0 means no edge defect. Random background failures also push some sectors over the
    threshold, so a defect-free wafer typically scores slightly above 0. Returns NaN without a wafer.
    """
    result = edge_sector_failures(image, empty_value, fail_value, band_width, num_sectors, sector_threshold)
    if result is None:
        return float("nan")
    mostly_failed, occupied = result
    return float(np.mean(mostly_failed[occupied]))


def edge_coverage_batch(images: np.ndarray, **kwargs: float) -> np.ndarray:
    return np.array([edge_coverage(image, **kwargs) for image in images])


def separability_auc(partial: np.ndarray, full: np.ndarray) -> float:
    """
    How well coverage alone separates the two labels: the probability that a random sample of the
    "full" class (e.g. Edge_Ring) has a higher coverage than a random sample of the "partial" class
    (e.g. Edge_Loc). This is the ROC AUC: 1.0 means perfectly separable, 0.5 no better than chance.
    """
    partial, full = partial[~np.isnan(partial)], full[~np.isnan(full)]
    scores = np.concatenate([partial, full])
    targets = np.concatenate([np.zeros(partial.size), np.ones(full.size)])
    return float(roc_auc_score(targets, scores))


def percentile_rank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Percentage (0-100) of `reference` values that are less than or equal to each value."""
    reference = np.sort(reference[~np.isnan(reference)])
    return 100.0 * np.searchsorted(reference, values, side="right") / reference.size
