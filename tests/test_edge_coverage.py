"""
Synthetic wafers: a disc of "pass" dies (1) on an empty background (0),
with failed dies (2) painted on part or all of the edge.
"""

import numpy as np
import pytest

from src.evaluation.edge_coverage import edge_band, edge_coverage, percentile_rank, separability_auc

SIZE = 52

# ================= Synthesised data =================


def wafer(radius: float = 24.0) -> np.ndarray:
    yy, xx = np.mgrid[:SIZE, :SIZE]
    image = np.zeros((SIZE, SIZE), dtype=np.uint8)
    image[(yy - 25.5) ** 2 + (xx - 25.5) ** 2 <= radius**2] = 1
    return image


# ================= Test: edge_band =================


def paint_edge(image: np.ndarray, start_deg: float, end_deg: float, width: int = 2) -> np.ndarray:
    """Mark the edge band as failed between two angles."""
    image = image.copy()
    yy, xx = np.mgrid[:SIZE, :SIZE]
    angles = np.degrees(np.arctan2(yy - 25.5, xx - 25.5)) % 360
    band = edge_band(image != 0, width)
    image[band & (angles >= start_deg) & (angles < end_deg)] = 2
    return image


def test_edge_band_is_a_thin_outer_ring():
    band = edge_band(wafer() != 0, width=2)
    assert band.any()
    assert not band[26, 26]  # the center is not part of the edge band
    assert band[26, 2]  # a pixel on the left edge of the disc is


# ================= Test: edge_coverage =================


def test_full_ring_scores_about_one():
    assert edge_coverage(paint_edge(wafer(), 0, 360)) == pytest.approx(1.0)


def test_local_edge_defect_scores_its_share_of_the_circumference():
    # 60 degrees is one sixth of the circumference, i.e. about 6 of 36 sectors
    assert edge_coverage(paint_edge(wafer(), 30, 90)) == pytest.approx(1 / 6, abs=2 / 36)


def test_clean_wafer_scores_zero():
    assert edge_coverage(wafer()) == 0.0


def test_random_background_failures_stay_well_below_a_ring():
    rng = np.random.default_rng(0)
    image = wafer()
    image[(image == 1) & (rng.random(image.shape) < 0.25)] = 2
    assert edge_coverage(image) < 0.3


def test_wafer_clipped_by_the_image_border():
    # A wafer larger than the image: pixels on the image border must count as wafer edge
    assert edge_coverage(paint_edge(wafer(radius=27), 0, 360)) == pytest.approx(1.0)


def test_empty_image_returns_nan():
    assert np.isnan(edge_coverage(np.zeros((SIZE, SIZE), dtype=np.uint8)))


# ================= Test: separability_auc =================


def test_separability_auc():
    assert separability_auc(np.array([0.1, 0.2, 0.3]), np.array([0.8, 0.9, 1.0])) == pytest.approx(1.0)
    assert separability_auc(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == pytest.approx(0.5)
    assert separability_auc(np.array([0.1, np.nan]), np.array([0.9])) == pytest.approx(1.0)


# ================= Test: percentile_rank =================


def test_percentile_rank():
    reference = np.array([0.1, 0.2, 0.3, 0.4])
    assert percentile_rank(np.array([0.05, 0.25, 0.4]), reference).tolist() == [0.0, 50.0, 100.0]
