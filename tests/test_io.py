from pathlib import Path

import numpy as np
import pytest

from src.data.io import load_npz_arrays

NUM_VALUES = 3

# ================= fixture: synthesised data =================


@pytest.fixture
def arrays() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    images = rng.integers(0, NUM_VALUES, size=(60, 8, 8)).astype(np.uint8)
    labels = np.tile(np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]]), (20, 1))
    return images, labels


# ================= Test: load_npz_arrays =================


def test_load_npz_arrays_downcasts_to_uint8(tmp_path: Path, arrays):
    images, labels = arrays
    path = tmp_path / "data.npz"
    np.savez(path, arr_0=images.astype(np.int64), arr_1=labels)
    loaded_images, loaded_labels = load_npz_arrays(path, "arr_0", "arr_1")
    assert loaded_images.dtype == np.uint8
    assert np.array_equal(loaded_images, images)
    assert np.array_equal(loaded_labels, labels)
