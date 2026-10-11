import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import DiscreteMapDataset, build_split_datasets, seed_worker
from src.data.split import split_indices
from src.data.transforms import RandomDihedral

NUM_VALUES = 3

# ================= fixture: synthesised data =================


@pytest.fixture
def arrays() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    images = rng.integers(0, NUM_VALUES, size=(60, 8, 8)).astype(np.uint8)
    labels = np.tile(np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]]), (20, 1))
    return images, labels


# ================= Test Cases =================


def test_item_shapes_and_dtypes(arrays):
    images, labels = arrays
    ds = DiscreteMapDataset(images, labels, np.arange(10), NUM_VALUES, encoding="onehot")
    x, y = ds[0]
    assert x.shape == (3, 8, 8) and x.dtype == torch.float32
    assert y.shape == (3,) and y.dtype == torch.float32


def test_single_label_targets_are_class_indices(arrays):
    images, _ = arrays
    labels_1d = np.arange(60) % 3
    ds = DiscreteMapDataset(images, labels_1d, np.arange(5), NUM_VALUES)
    _, y = ds[4]
    assert y.dtype == torch.long and y.item() == 1


def test_out_of_range_indices_raise(arrays):
    images, labels = arrays
    with pytest.raises(ValueError, match="out of range"):
        DiscreteMapDataset(images, labels, np.array([0, 60]), NUM_VALUES)


def test_split_datasets_only_augment_train(arrays):
    images, labels = arrays
    split = split_indices(labels, val_ratio=0.2, test_ratio=0.2, seed=0, min_stratum_count=2)
    datasets = build_split_datasets(images, labels, split, NUM_VALUES, train_transform=RandomDihedral(seed=0))

    assert {name: len(ds) for name, ds in datasets.items()} == split.sizes()
    assert datasets["train"].transform is not None

    # val/test are deterministic: the same item always comes back identical
    for name in ("val", "test"):
        assert datasets[name].transform is None
        assert torch.equal(datasets[name][0][0], datasets[name][0][0])


def test_workers_get_independent_augmentation_streams():
    # Every item is the same asymmetric map, so any difference between items comes from augmentation.
    base = np.array(
        [
            [0, 1, 1, 2],
            [0, 0, 1, 1],
            [2, 0, 0, 1],
            [1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )

    images = np.repeat(base[np.newaxis], 32, axis=0)
    labels = np.zeros((32, 3))
    ds = DiscreteMapDataset(images, labels, np.arange(32), NUM_VALUES, transform=RandomDihedral(seed=0))

    torch.manual_seed(0)
    # "spawn" starts each work as a fresh process.
    # Avoid the fork() deadlock risk and also provice the dataset and seed_worker are picklable.
    loader = DataLoader(ds, batch_size=1, num_workers=2, worker_init_fn=seed_worker, multiprocessing_context="spawn")
    outputs = [x[0].numpy().tobytes() for x, _ in loader]

    # With batch_size=1 and 2 workers, items alternate between workers.
    # Without seed_worker both workers would start from the same copied RNG state and produce identical sequences.
    assert outputs[0::2] != outputs[1::2]
