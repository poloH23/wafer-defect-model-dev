# Encoding andaugmentation for images whole pixel values are discrete categories, e.g. wafer map (0=empty, 1=pass, 2=defect).

from typing import Literal
import numpy as np
EncodingMode = Literal["onehot", "scaled"]

def encode_discrete(image: np.ndarray, num_values: int, mode: EncodingMode = "onehot") -> np.ndarray:
    """
    Convert a 2D map of category ids (H, W) into a float32 model input (C, H, W).

    - "onehot": one channel per category (C = num_values). Treate pixel values as categories rather than intensities.
    - "scaled": a single channel holding value / (num_values - 1). Impose an order and a distance between categories.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image (H, W), got shape {image.shape}")
    if num_values < 2:
        raise ValueError("num_value must be at least 2")
    if image.min() < 0 or image.max() >= num_values:
        raise ValueError(f"Pixel values must be within [0, {num_values - 1}], got [{image.min()}, {image.max()}]")

    if mode == "onehot":
        categories = np.arange(num_values).reshape(-1, 1, 1)
        return (image[np.newaxis] == categories).astype(np.float32)
    if mode == "scaled":
        return (image[np.newaxis] / (num_values - 1)).astype(np.float32)
    raise ValueError(f"Unsupported encoding mode: {mode}")

def encoded_channels(num_values: int, mode: EncodingMode = "onehot") -> int:
    """Number of channel encode_discrete produces, used to sized the model's first layer."""
    if mode == "onehot":
        return num_values
    if mode == "scaled":
        return 1
    raise ValueError(f"Unsupported encoding mode: {mode}")


def apply_dihedral(image: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """
    Apply one of the 8 symmetries of a square (dihedral group D4) ot the last two axes.

    Rotation by multiples of 90 degrees and flips only move pixels and never interpolate them,
    so discrete values stay exactly whithin the original set. Works on (H, W) and (C, H, W).

    Returns a contiguous copy: np.rot90 / np.flip return views with negative strides,
    which torch.from_numpy cannot handle.
    """
    out = np.rot90(image, k=k % 4, axes=(-2, -1))
    if flip:
        out = np.flip(out, axis=-1)
    return np.ascontiguousarray(out)

class RandomDihedral:
    """
    Randomly apply one of the 8 D4 symmetries, each with equal probabillity (identity included).

    Only valid for when orientation carries no meaning (wafer maps, microscopy, aerial images).
    Do not use it where orientation matters (digits such as 6 and 9, text, scense with a sky).

    Randomness comes from a private np.random.Generator. A multi-worker DataLoader gives every worker an identical copy of this project.
    src.data.dataset.seed_worker reseeds it per work to prevent all worker get the same "random" sequence.
    """
    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        k = int(self._rng.integers(4))
        flip = bool(self._rng.integers(2))
        return apply_dihedral(image, k, flip)
