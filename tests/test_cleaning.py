import numpy as np
import pytest

from src.data.cleaning import apply_invalid_pixel_policy, impute_neighbor_mode

NUM_VALUES = 3


def blank(value: int = 1) -> np.ndarray:
    return np.full((1, 5, 5), value, dtype=np.uint8)


# ================= Test: apply_invalid_pixel_policy =================


def test_clean_data_is_returned_unchanged():
    images = blank()
    cleaned, report = apply_invalid_pixel_policy(images, NUM_VALUES, policy="error")
    assert cleaned is images
    assert report["invalid_pixels"] == 0


def test_error_policy_reports_statistics():
    images = np.concatenate([blank(), blank(), blank()])
    images[0, 2, 2] = 3
    images[2, 0, 0] = 3
    images[2, 4, 4] = 3
    with pytest.raises(ValueError, match=r"2 images contain 3 pixels outside \[0, 2\] \(values \[3\]\). Set invalid_pixel_policy in the dataset config to handle them."):
        apply_invalid_pixel_policy(images, NUM_VALUES, policy="error")


def test_neighbor_mode_policy_report():
    images = np.concatenate([blank(2), blank(0)])
    images[0, 2, 2] = 3
    images[1, 2, 2] = 3
    _, report = apply_invalid_pixel_policy(images, NUM_VALUES, policy="neighbor_mode")
    assert report == {"policy": "neighbor_mode", "invalid_pixels": 2, "invalid_images": 2, "invalid_values": [3], "replaced_with": {"0": 1, "2": 1}}


# ================= Test: impute_neighbor_mode =================


def test_pixel_takes_the_value_of_its_surroundings():
    images = blank(2)
    images[0, 2, 2] = 3
    cleaned, replaced = impute_neighbor_mode(images, NUM_VALUES)
    assert cleaned[0, 2, 2] == 2
    assert replaced == {2: 1}


def test_corner_pixel_uses_the_clipped_window():
    images = blank(1)
    images[0, 0, 1] = images[0, 1, 0] = images[0, 1, 1] = 0
    images[0, 0, 0] = 3
    cleaned, _ = impute_neighbor_mode(images, NUM_VALUES)
    assert cleaned[0, 0, 0] == 0  # its only 3 neighbors are all 0


def test_adjacent_invalid_pixels_are_all_imputed():
    images = blank(2)
    images[0, 2, 1:4] = 3
    cleaned, replaced = impute_neighbor_mode(images, NUM_VALUES)
    assert (cleaned < NUM_VALUES).all()
    assert replaced == {2: 3}


def test_ties_go_to_the_smallest_value():
    images = np.array([[[1, 1, 1], [2, 3, 2], [2, 2, 1]]], dtype=np.uint8)  # 4 ones, 4 twos around the center
    cleaned, _ = impute_neighbor_mode(images, NUM_VALUES)
    assert cleaned[0, 1, 1] == 1


def test_input_is_not_modified():
    images = blank()
    images[0, 2, 2] = 3
    impute_neighbor_mode(images, NUM_VALUES)
    assert images[0, 2, 2] == 3


def test_no_valid_neighbor_raises():
    images = np.full((1, 3, 3), 3, dtype=np.uint8)
    with pytest.raises(ValueError, match="none of its neighbors"):
        impute_neighbor_mode(images, NUM_VALUES)
