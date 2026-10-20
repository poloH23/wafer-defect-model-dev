# PyTorch Dataset for images whole pixel values are discrete categories.
# All encoding and augmentation logic lives in the NumPy-only module (src.data.transforms).
# This module only selects samples by index and converts them to tensors.


import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from src.data.split import SplitResult
from src.data.transforms import EncodingMode, RandomDihedral, encode_discrete


class DiscreteMapDataset(Dataset):
    """
    Serves (image, target) pair for the samples selected by 'indices'.
    - image: float32 tensor (C, H, W), produced by encode_discrete, then augmented if a transform is given
    - target: float32 multi-hot vector for 2D labels (EncodingMode),
              int64 class index for 1D labels (CrossEntropyLoss).
    """
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        num_values: int,
        encoding: EncodingMode = "onehot",
        transform: RandomDihedral | None = None
    ):
        if len(images) != len(labels):
            raise ValueError(f"Images and labels differ in length: {len(images)} vs {len(labels)}")
        indices = np.asarray(indices)
        if indices.size and (indices.min() < 0 or indices.max() >= len(images)):
            raise ValueError(f"Split indices out of range for the dataset of {len(images)} samples")

        self.images = images
        self.labels = labels
        self.indices = indices
        self.num_values = num_values
        self.encoding = encoding
        self.transform = transform

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.indices[i]
        image = encode_discrete(self.images[idx], self.num_values, self.encoding)
        if self.transform is not None:
            image = self.transform(image)
        return torch.from_numpy(np.ascontiguousarray(image)), self._target(idx)

    def _target(self, idx: int) -> torch.Tensor:
        if self.labels.ndim == 1:
            # Class index, as expected by CrossEntropy
            return torch.tensor(int(self.labels[idx]), dtype=torch.long)
        # Multi-hot vector, as expected by BCEWithLogitsLoss
        return torch.from_numpy(np.asarray(self.labels[idx], dtype=np.float32))

def build_split_datasets(
    images: np.ndarray,
    labels: np.ndarray,
    split: SplitResult,
    num_values: int,
    encoding: EncodingMode = "onehot",
    train_transform: RandomDihedral | None = None,
) -> dict[str, DiscreteMapDataset]:
    """Augmentation is applied to the training split only; val/test always see the original images."""
    return {
        "train": DiscreteMapDataset(images, labels, split.train_idx, num_values, encoding, train_transform),
        "val": DiscreteMapDataset(images, labels, split.val_idx, num_values, encoding, None),
        "test": DiscreteMapDataset(images, labels, split.test_idx, num_values, encoding, None)
    }

def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker_init_fn: give each worker's augmentation its own stream.

    Usage:
        DataLoader(dataset, num_workers=4, worker_init_fn=seed_worker, ...)
    """
    info = get_worker_info()
    if info is None:
        return
    transform = getattr(info.dataset, "transform", None)
    if transform is not None and hasattr(transform, "reseed"):
        transform.reseed(torch.initial_seed() % 2**32)
