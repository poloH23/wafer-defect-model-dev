# from src.common.path import setup_python_path

# setup_python_path()
from pathlib import Path

import numpy as np
import pytest

from src.eda.npz_analyzer import NPZAnalyzer


@pytest.fixture
def mock_single_label_npz(tmp_path: Path):
    """Fixture: Create a temporary single-label virtual NPZ archive."""
    # Generate 10 virtual images (5x5) and 10 1D labels (category 0-2)
    images = np.random.randint(0, 255, size=(10, 5, 5))
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 2])

    # Save to the temporary path provided by pytest.
    file_path = tmp_path / "mock_single.npz"
    np.savez(file_path, arr_0=images, arr_1=labels)
    return str(file_path)


@pytest.fixture
def mock_multi_label_npz(tmp_path: Path):
    """Fixture: Create a temporary multi-label virtual NPZ archive."""
    # Generate 10 virtual images (5x5) and 10 2D labels (category 0-2)
    images = np.random.randint(0, 255, size=(10, 5, 5))
    labels = np.random.randint(0, 2, size=(10, 4))

    # Save to the temporary path provided by pytest.
    file_path = tmp_path / "mock_multi.npz"
    np.savez(file_path, arr_0=images, arr_1=labels)
    return str(file_path)


# ================= Test Cases =================


def test_analyzer_with_single_label(mock_single_label_npz: Path):
    """Test whether the parser correctly handles 1D labels."""
    analyzer = NPZAnalyzer(npz_path=mock_single_label_npz, image_key="arr_0", label_key="arr_1")

    assert analyzer.num_samples == 10
    assert analyzer.image_shape == (5, 5)
    assert analyzer.label_mode == "single-label"
    assert analyzer.num_classes == 3

    dist = analyzer.get_distribution()
    assert dist["class_0"] == 4


def test_analyzer_with_multi_label(mock_multi_label_npz: Path):
    """Test whether the parser correctly handles 2D labels."""
    analyzer = NPZAnalyzer(npz_path=mock_multi_label_npz, image_key="arr_0", label_key="arr_1")

    assert analyzer.image_shape == (5, 5)
    assert analyzer.num_classes == 4
    assert "multi-label" in analyzer.label_mode or "one-hot-single-label" in analyzer.label_mode
