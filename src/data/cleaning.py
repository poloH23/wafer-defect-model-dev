# Policies for pixel values outside the documented range of a discrete-valued dataset.
# The raw data is never modified: cleaning is applied in memery after loading, so the
# file hash stays valid, and the rule is visible in code and config.

from collections import Counter
from typing import Any, Literal

import numpy as np

InvalidPixelPolicy = Literal["error", "neighbor_mode"]

def out_of_range_mask(images: np.ndarray, num_values: int) -> np.ndarray:
    """True whether a pixel is outside [0, num_values - 1]."""
    return (images < 0) | (images >= num_values)

def impute_neighbor_mode(images: np.ndarray, num_values: int) -> tuple[np.ndarray, dict[int, int]]:
    """
    Replace every out-range pixel with the most common valid value among its 8 neighbors
    (fewer at the image border). Ties go to the smallest value.

    Pixels are processed in order, so a neighbor imputed earlier counts as valid; this handles
    small clusters of adjacent invalid pixels. Returns a cleaned copy (the input is not modified)
    and a {new value: number of pixels} summary.
    """
    cleaned = images.copy()
    replaced: Counter[int] = Counter()
    height, width = images.shape[-2:]

    for n, i, j in zip(*np.nonzero(out_of_range_mask(images, num_values)), strict=True):
        window = cleaned[n, max(i - 1, 0) : min(i + 2, height), max(j - 1, 0) : min(j + 2, width)]
        valid = window[~out_of_range_mask(window, num_values)] # the pixel itself is still invalid, need to be excluded
        if valid.size == 0:
            raise ValueError(f"Cannot impute pixel({n}, {i}, {j}): none of its neighbors has a valid value")
        value = int(np.bincount(valid.astype(np.int64), minlength=num_values).argmax())
        cleaned[n, i, j] = value
        replaced[value] += 1
    return cleaned, dict(sorted(replaced.items()))

def apply_invalid_pixel_policy(images: np.ndarray, num_values: int, policy: InvalidPixelPolicy = "error") -> tuple[np.ndarray, dict[str, Any]]:
    """
    Check the whole dataset at once, right after loading, so a data problem fails within seconds
    with full statistics instead of in the middle of the first training epoch.

    - "error": raise if any pixel is out of range
    - "neighbor_mode": impute out-of-range pixels with impute_neighbor_mode

    Returns the (possibly cleaned) images and a report to record with the run.
    """
    mask = out_of_range_mask(images, num_values)
    num_pixels = int(mask.sum())
    report: dict[str, Any] = {"policy": policy, "invalid_pixels": num_pixels, "invalid_images": 0, "replaced_with": {}}
    if num_pixels == 0:
        return images, report

    report["invalid_images"] = int(mask.any(axis=tuple(range(1, images.ndim))).sum())
    invalid_values = np.unique(images[mask]).tolist()
    if policy == "error":
        raise ValueError(
            f"{report['invalid_images']} images contain {num_pixels} pixels outside [0, {num_values - 1}] "
            f"(values {invalid_values}). Set invalid_pixel_policy in the dataset config to handle them."
        )
    if policy != "neighbor_mode":
        raise ValueError(f"Unsupported invalid pixel policy: {policy}")

    cleaned, replaced = impute_neighbor_mode(images, num_values)
    report["invalid_values"] = invalid_values
    report["replaced_with"] = {str(value): count for value, count in replaced.items()}
    return cleaned, report
