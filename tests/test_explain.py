import math

import numpy as np
import pytest
import torch

from src.evaluation.explain import angular_error, cam_target_layer, compute_cams, localization_ratio, pointing_hit, rank_correlation, wafer_regions
from src.models.factory import SUPPORTED_MODELS, build_model

SIZE = 52


def wafer(radius: float = 24.0) -> np.ndarray:
    yy, xx = np.mgrid[:SIZE, :SIZE]
    image = np.zeros((SIZE, SIZE), dtype=np.uint8)
    image[(yy - 25.5) ** 2 + (xx - 25.5) ** 2 <= radius**2] = 1
    return image


@pytest.mark.parametrize("name", SUPPORTED_MODELS)
def test_every_architecture_has_a_spatial_cam_layer(name):
    model = build_model(name, num_classes=8, in_channels=3).eval()
    captured = {}
    cam_target_layer(model, name).register_forward_hook(lambda _m, _i, output: captured.update(size=output.shape[-1]))
    with torch.no_grad():
        model(torch.zeros(1, 3, SIZE, SIZE))
    assert captured["size"] == 7


def test_compute_cams_returns_one_normalized_map_per_input():
    torch.manual_seed(0)
    model = build_model("shufflenet_v2_x1_0", num_classes=8, in_channels=3).eval()
    cams = compute_cams(model, cam_target_layer(model, "shufflenet_v2_x1_0"), torch.rand(3, 3, SIZE, SIZE), classes=[0, 2, 5])
    assert cams.shape == (3, SIZE, SIZE)
    assert cams.min() >= 0.0 and cams.max() <= 1.0


def test_zones_partition_the_wafer_where_their_names_say():
    regions = wafer_regions(wafer())
    assert regions["center"][26, 26] and regions["ring"][26, 12] and regions["edge"][26, 2]
    total = regions["center"].astype(int) + regions["ring"].astype(int) + regions["edge"].astype(int)
    assert np.array_equal(total, regions["wafer"].astype(int))  # every wafer pixel in exactly one zone


def test_localization_ratio():
    regions = wafer_regions(wafer())
    wafer_mask, edge = regions["wafer"], regions["edge"]
    area_share = edge.sum() / wafer_mask.sum()
    assert localization_ratio(np.ones((SIZE, SIZE)), edge, wafer_mask) == pytest.approx(1.0)  # no focus
    assert localization_ratio(edge.astype(float), edge, wafer_mask) == pytest.approx(1 / area_share)  # all mass on the edge
    assert localization_ratio(regions["center"].astype(float), edge, wafer_mask) == 0.0
    assert math.isnan(localization_ratio(np.zeros((SIZE, SIZE)), edge, wafer_mask))


def test_rank_correlation():
    mask = wafer() != 0
    a = np.random.default_rng(0).random((SIZE, SIZE))
    assert rank_correlation(a, a, mask) == pytest.approx(1.0)
    assert rank_correlation(a, -a, mask) == pytest.approx(-1.0)
    assert math.isnan(rank_correlation(a, np.zeros((SIZE, SIZE)), mask))


def test_pointing_hit():
    region = wafer_regions(wafer())["center"]
    cam = np.zeros((SIZE, SIZE))
    cam[26, 26] = 1.0
    assert pointing_hit(cam, region) == 1.0
    cam[26, 26], cam[26, 2] = 0.0, 1.0
    assert pointing_hit(cam, region) == 0.0
    assert math.isnan(pointing_hit(np.zeros((SIZE, SIZE)), region))


def test_angular_error_measures_where_the_heatmap_points_along_the_edge():
    image = wafer()
    yy, xx = np.mgrid[:SIZE, :SIZE]
    angles = np.degrees(np.arctan2(yy - 25.5, xx - 25.5)) % 360
    edge = wafer_regions(image)["edge"]
    image[edge & (angles < 40)] = 2  # a local defect on the edge, around 20 degrees
    toward = np.exp(-((yy - 30) ** 2 + (xx - 40) ** 2) / 20.0)  # a blob one cell inward, in that direction
    away = np.exp(-((yy - 21) ** 2 + (xx - 11) ** 2) / 20.0)  # the same blob on the opposite side
    assert angular_error(toward, image) < 15
    assert angular_error(away, image) > 165
    assert math.isnan(angular_error(np.zeros((SIZE, SIZE)), image))
