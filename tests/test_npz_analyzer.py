from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eda.npz_analyzer import NPZAnalyzer

# ================= fixture: synthesised data =================


@pytest.fixture
def npz_single_label(tmp_path: Path):
    """5 samples with 1D sigle label (class 1/1/2 at least appears once)."""
    images = np.random.randint(0, 3, size=(5, 8, 8)).astype(np.uint8)
    labels = np.array([0, 1, 2, 1, 0])
    path = tmp_path / "single_label.npz"
    np.savez(path, images=images, labels=labels)
    return path


@pytest.fixture
def npz_one_hot(tmp_path: Path):
    """4 samples with one-hot single label."""
    images = np.random.randint(0, 3, size=(4, 8, 8)).astype(np.uint8)
    labels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    path = tmp_path / "one_hot.npz"
    np.savez(path, images=images, labels=labels)
    return path


@pytest.fixture
def npz_multi_label(tmp_path: Path):
    """ "6 samples with multi-label: single-label, mixed-label, normal"""
    images = np.random.randint(0, 3, size=(6, 8, 8)).astype(np.uint8)
    labels = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 0, 0]])
    path = tmp_path / "multi_label.npz"
    np.savez(path, images=images, labels=labels)
    return path


# ================= __init__ =================


class TestInit:
    def test_valid_file_single_label(self, npz_single_label):
        analyzer = NPZAnalyzer(npz_single_label, image_key="images", label_key="labels")
        assert analyzer.num_samples == 5
        assert analyzer.image_shape == (8, 8)
        assert analyzer.label_mode == "single-label"
        assert analyzer.num_classes == 3

    def test_valid_file_one_hot(self, npz_one_hot):
        analyzer = NPZAnalyzer(npz_one_hot, image_key="images", label_key="labels")
        assert analyzer.label_mode == "one-hot-single-label"
        assert analyzer.num_classes == 3

    def test_valid_file_multi_label(self, npz_multi_label):
        analyzer = NPZAnalyzer(npz_multi_label, image_key="images", label_key="labels")
        assert "multi-label" in analyzer.label_mode
        assert analyzer.num_classes == 3

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            NPZAnalyzer(tmp_path / "nonexistent.npz", image_key="images", label_key="labels")

    def test_wrong_suffix(self, tmp_path):
        # If the file extension is not npz, raising FileNotFoundError
        fake = tmp_path / "data.txt"
        fake.write_text("not an npz")
        with pytest.raises(FileNotFoundError):
            NPZAnalyzer(fake, image_key="images", label_key="labels")

    def test_wrong_keys(self, npz_single_label):
        with pytest.raises(ValueError, match="Wrong keys"):
            NPZAnalyzer(npz_single_label, image_key="wrong_key", label_key="labels")

    def test_invalid_label_ndim_raises(self, tmp_path):
        images = np.zeros((2, 4, 4), dtype=np.uint8)
        labels = np.zeros((2, 2, 2))  # 3D，_analyze_label_structure 不支援
        path = tmp_path / "bad_ndim.npz"
        np.savez(path, images=images, labels=labels)
        with pytest.raises(ValueError, match="Unsupported label dimension"):
            NPZAnalyzer(path, image_key="images", label_key="labels")


# ================= get_distribution =================


class TestDistribution:
    def test_single_label_distribution(self, npz_single_label):
        analyzer = NPZAnalyzer(npz_single_label, image_key="images", label_key="labels")
        dist = analyzer.get_distribution()
        assert dist == {"class_0": 2, "class_1": 2, "class_2": 1}
        for v in dist.values():
            assert isinstance(v, int)


# ================= create_sample_polot / show_sample / save_sample =================


class TestSamplePlot:
    def test_returns_figure(self, npz_single_label):
        analyzer = NPZAnalyzer(npz_single_label, image_key="images", label_key="labels")
        fig = analyzer.create_sample_plot(0)
        assert isinstance(fig, plt.Figure)  # type: ignore
        plt.close(fig)

    @pytest.mark.parametrize("bad_index", [-1, 5, 100])
    def test_index_out_of_range(self, npz_single_label, bad_index):
        analyzer = NPZAnalyzer(npz_single_label, image_key="images", label_key="labels")
        with pytest.raises(IndexError):
            analyzer.create_sample_plot(bad_index)

    def test_channel_first_image_does_not_crash(self, tmp_path):
        # (N, C, H, W) format，C=3
        images = np.random.randint(0, 255, size=(2, 3, 8, 8)).astype(np.uint8)
        labels = np.array([0, 1])
        path = tmp_path / "chw.npz"
        np.savez(path, images=images, labels=labels)
        analyzer = NPZAnalyzer(path, image_key="images", label_key="labels")
        fig = analyzer.create_sample_plot(0)
        assert isinstance(fig, plt.Figure)  # type: ignore
        plt.close(fig)

    def test_save_sample_creates_file(self, npz_single_label, tmp_path):
        analyzer = NPZAnalyzer(npz_single_label, image_key="images", label_key="labels")
        save_path = tmp_path / "nested" / "sample.png"
        result = analyzer.save_sample(0, save_path)
        assert save_path.exists()
        assert result == save_path

    def test_save_sample_none_raises(self, npz_single_label):
        analyzer = NPZAnalyzer(npz_single_label, image_key="images", label_key="labels")
        with pytest.raises(ValueError):
            analyzer.save_sample(0, None)  # type: ignore


# ================= get_class_representative_indices =================


class TestClassRepresentatives:
    def test_single_label_one_per_class(self, npz_single_label):
        analyzer = NPZAnalyzer(npz_single_label, image_key="images", label_key="labels")
        reps = analyzer.get_class_representative_indices()
        assert reps == {"class_0": 0, "class_1": 1, "class_2": 2}

    def test_multi_label_prefers_pure_sample(self, npz_multi_label):
        analyzer = NPZAnalyzer(npz_multi_label, image_key="images", label_key="labels")
        reps = analyzer.get_class_representative_indices(prefer_pure=True)
        # inx 0/1/3 are single label examples, they should be chosen before idx 2 and 4
        assert reps["class_0"] == 0
        assert reps["class_1"] == 1
        assert reps["class_2"] == 3

    def test_falls_back_to_impure_when_no_pure_sample(self, tmp_path):
        images = np.zeros((2, 4, 4), dtype=np.uint8)
        labels = np.array([[1, 1], [1, 1]])  # mixed-label sample only
        path = tmp_path / "no_pure.npz"
        np.savez(path, images=images, labels=labels)
        analyzer = NPZAnalyzer(path, image_key="images", label_key="labels")
        reps = analyzer.get_class_representative_indices(prefer_pure=True)
        assert reps["class_0"] == 0
        assert reps["class_1"] == 0

    def test_missing_class_skipped(self, tmp_path):
        images = np.zeros((2, 4, 4), dtype=np.uint8)
        labels = np.array([[1, 0], [1, 0]])  # class_1 has never shown
        path = tmp_path / "missing_class.npz"
        np.savez(path, images=images, labels=labels)
        analyzer = NPZAnalyzer(path, image_key="images", label_key="labels")
        reps = analyzer.get_class_representative_indices()
        assert "class_1" not in reps

    # normal (only 0 label) sample has not been handled by 'get_class_representative_indices'
    def test_normal_sample_included(self, npz_multi_label):
        analyzer = NPZAnalyzer(npz_multi_label, image_key="images", label_key="labels")
        reps = analyzer.get_class_representative_indices()
        assert "normal" in reps
