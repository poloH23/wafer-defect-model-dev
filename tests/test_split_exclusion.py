import numpy as np
import pytest

from src.data.split import EXCLUDED_STRATUM, load_split, save_split, split_digest, split_indices, split_report

# ================= fixture: synthesised project root =================


@pytest.fixture
def labels() -> np.ndarray:
    return np.tile(np.array([[1, 0], [0, 1], [1, 1]]), (100, 1))


def test_excluded_samples_appear_in_no_split(labels):
    exclude = np.arange(0, 300, 7)
    result = split_indices(labels, seed=0, exclude=exclude)
    used = np.concatenate([result.train_idx, result.val_idx, result.test_idx])
    assert np.intersect1d(used, exclude).size == 0
    assert used.size + exclude.size == len(labels)
    assert (result.strata[exclude] == EXCLUDED_STRATUM).all()


# ================= Test Cases =================


def test_empty_exclusion_matches_no_exclusion(labels):
    # Guarantees that adding the option did not change splits made without it
    assert split_digest(split_indices(labels, seed=0)) == split_digest(split_indices(labels, seed=0, exclude=np.array([], dtype=int)))


def test_report_ignores_excluded_samples(labels):
    result = split_indices(labels, seed=0, exclude=np.arange(10))
    assert all(row["stratum"] != EXCLUDED_STRATUM for row in split_report(result))
    assert sum(row["total"] for row in split_report(result)) == len(labels) - 10


def test_out_of_range_exclusion_raises(labels):
    with pytest.raises(ValueError, match="out of range"):
        split_indices(labels, seed=0, exclude=np.array([len(labels)]))


def test_excluded_indices_survive_save_and_load(labels, tmp_path):
    result = split_indices(labels, seed=0, exclude=np.array([3, 5, 8]))
    save_split(result, tmp_path)
    assert load_split(tmp_path).excluded_idx.tolist() == [3, 5, 8]


def test_split_saved_without_exclusion_field_still_loads(labels, tmp_path):
    result = split_indices(labels, seed=0)
    np.savez_compressed(tmp_path / "split_indices.npz", train=result.train_idx, val=result.val_idx, test=result.test_idx, strata=result.strata)
    assert load_split(tmp_path).excluded_idx.size == 0
