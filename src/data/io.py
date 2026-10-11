
from pathlib import Path
import numpy as np

def load_npz_arrays(path: str | Path, image_key: str, label_key: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load images and labels once, so every split shares the same arrays instead of copying them.
    Integer images whose values fit in uint8 are stored as uint8 for memory saving.
    """
    with np.load(path) as data:
        images = data[image_key]
        labels = data[label_key]

    if np.issubdtype(images.dtype, np.integer) and images.min() >= 0 and images.max() <= np.iinfo(np.uint8).max:
        images = images.astype(np.uint8)
    return images, labels
