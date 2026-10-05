import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

RARE_STRATUM = -1
SPLIT_FILE = "split_indices.npz"
META_FILE = "split_meta.json"


@dataclass(frozen=True)
class SplitResult:
    """Indices of each split, plus the stratum id assigned to every sample."""
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    strata: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {"train": int(self.train_idx.size), "val": int(self.val_idx.size), "test": int(self.test_idx.size)}


def make_strata(labels: np.ndarray, min_count: int = 10) -> np.ndarray:
    """
    Convert labels into stratum ids used for stratified splitting.

    - 1D labels (single-label): each class value is its own stratum.
    - 2D labels (one-hot / multi-label): each exac label combination is its own stratum.
      so the proportion of every combination is preserved across split.

    Strata with fewer than `min_count` samples are merged into a single RARE_STRATUM,
    because scikit-learn cannot stratify a group that is too small to appear in every split.
    """
    if labels.ndim == 1:
        _, inverse = np.unique(labels, return_inverse=True)
    elif labels.ndim == 2:
        _, inverse = np.unique(labels, axis=0, return_inverse=True)
    else:
        raise ValueError(f"Unsupported label dimension: {labels.shape}")

    # revel() guards against NumPy version that return the inverse with an extra axis;
    # astype() returns a copy, so motifying it below never touches the caller's data
    strata = inverse.ravel().astype(np.int64)

    counts = np.bincount(strata)
    rare_mask = counts[strata] < min_count
    if rare_mask.any():
        strata[rare_mask] = RARE_STRATUM
    return strata

def split_indices(
    labels: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    min_stratum_count: int = 10
) -> SplitResult:
    """
    Split samples indices into train/val/test with stratification.

    The split is done in two steps:
    1) Hold out the test set first
    2) Then split the validation set from the remaining samples.
    The ratio is rescaled so that `val_ratio` refers to the propotion of the whole dataset.

    Returned indices are sorted, which keeps reads sequential for NPZ/HDF5 sources.
    """
    if any([val_ratio <= 0, test_ratio <= 0, val_ratio + test_ratio >= 1.0]):
        raise ValueError("Require val_ratio > 0, test_ratio > 0 and val_ratio + test_ratio < 1")

    num_samples = labels.shape[0]
    all_idx = np.arange(num_samples)
    strata = make_strata(labels, min_count=min_stratum_count)

    _, strata_counts = np.unique(strata, return_counts=True)
    if strata_counts.min() < 2:
        raise ValueError("A stratum has fewer than 2 samples even after merging rare stratas; increase min_strata_count or check the labels.")

    # Step 1: hold out the test set
    trainval_idx, test_idx = train_test_split(all_idx, test_size=test_ratio, stratify=strata, random_state=seed, shuffle=True)

    # Step 2: split the validation set from remaining samples
    relative_val_ratio = val_ratio / (1.0 - test_ratio)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=relative_val_ratio, stratify=strata[trainval_idx], random_state=seed, shuffle=True)

    result = SplitResult(train_idx=np.sort(train_idx), val_idx=np.sort(val_idx), test_idx=np.sort(test_idx), strata=strata)
    _check_partition(result, num_samples)
    return result

def _check_partition(result: SplitResult, num_samples: int) -> None:
    """Guarantee the three splits and disjoint and together cover every sample exactly once."""
    combined = np.concatenate([result.train_idx, result.val_idx, result.test_idx])
    if combined.size != num_samples or np.unique(combined).size != num_samples:
        raise RuntimeError("Split indices overlap or do not cover every sample.")

def split_digest(result: SplitResult) -> str:
    """
    Content fingerprint of the split itself (not of the source data).
    Two splits get the same digest only if every sample lands in the same subset.
    Training runs record this value, so any result can be tracked to the exact split it used.
    """
    digest = hashlib.sha256()
    digest.update(np.array(list(result.sizes().values()), dtype=np.int64).tobytes())
    for idx in (result.train_idx, result.val_idx, result.test_idx):
        digest.update(np.ascontiguousarray(idx, dtype=np.int64).tobytes())
    return digest.hexdigest()

def split_report(result: SplitResult) -> list[dict[str, int]]:
    """Per-stratum sample counts in each split, used to vertify the stratification."""
    split_strata = {
        "train": result.strata[result.train_idx],
        "val": result.strata[result.val_idx],
        "test": result.strata[result.test_idx]
    }
    rows = []
    for stratum in np.unique(result.strata):
        row = {"stratum": int(stratum), "total": int(np.sum(result.strata == stratum))}
        for name, values in split_strata.items():
            row[name] = int(np.sum(values == stratum))
        rows.append(row)
    return rows

def save_split(result: SplitResult, output_dir: str | Path, metadata: dict[str, Any] | None = None, overwrite: bool = False) -> Path:
    """Save split indices (NPZ) and metadata (JSON)."""
    output_dir = Path(output_dir)
    split_path = output_dir / SPLIT_FILE
    if split_path.exists() and not overwrite:
        raise FileExistsError(f"Split already exists: {split_path}. Use a new split name, or pass overwrite=True If you are sure no experiment depends on it.")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(split_path, train=result.train_idx, val=result.val_idx, test=result.test_idx, strata=result.strata)

    meta = {
        "split_digest": split_digest(result),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sizes": result.sizes(),
        "num_strata": int(np.unique(result.strata).size),
        "library_version": {"numpy": np.__version__, "scikit-learn": sklearn.__version__},
        **(metadata or {})
    }
    with open(output_dir / META_FILE, 'w', encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return split_path

def load_split(split_dir: str | Path) -> SplitResult:
    """Load split indices saved by save_split()."""
    split_path = Path(split_dir) / SPLIT_FILE
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with np.load(split_path) as data:
        return SplitResult(train_idx=data["train"], val_idx=data["val"], test_idx=data["test"], strata=data["strata"])


def load_split_meta(split_dir: str | Path) -> dict[str, Any]:
    """Load the metadata saved next to the split indices (digest, source hash, parameters)."""
    meta_path = Path(split_dir) / META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"Split metadata not found: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)
