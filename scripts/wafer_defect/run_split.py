import hashlib
from pathlib import Path

import numpy as np

from src.common.path import get_project_root
from src.data.split import save_split, split_indices, split_report

# ================= Project-specific Settings =================

LABEL_KEY = "arr_1"
SEED = 42
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MIN_STRATUM_COUNT = 10
SPLIT_NAME = "v1"

PROJECT_ROOT = get_project_root()
DATA_SOURCE = PROJECT_ROOT / "data" / "raw" / "Wafer_Map_Datasets.npz"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits" / "wafer_defect" / SPLIT_NAME
SAVE_OVERWRITE = False


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Fingerprint of the source file, so the split can be tied to the exact data it was made from."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    # Only the label array of the NPZ data is loaded here, the images are never loaded.
    with np.load(DATA_SOURCE) as data:
        labels = data[LABEL_KEY]

    result = split_indices(labels, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=SEED, min_stratum_count=MIN_STRATUM_COUNT)
    print(f"Split sizes: {result.sizes()}")

    # Stratification check: how far each stratum's val/test share drifts from the target ratio
    rows = split_report(result)
    max_val_dev = max(abs(r["val"] / r["total"] - VAL_RATIO) for r in rows)
    max_test_dev = max(abs(r["test"] / r["total"] - TEST_RATIO) for r in rows)
    smallest = min(rows, key=lambda r: r["total"])
    print(f"Max pre-stratum deviation -> val: {max_val_dev:.3%}, test: {max_test_dev:.3%}")
    print(f"Smallest stratum -> total: {smallest['total']}, train: {smallest['train']}, val: {smallest['val']}, test: {smallest['test']}")

    split_path = save_split(
        result=result,
        output_dir=SPLIT_DIR,
        metadata={
            "source_file": DATA_SOURCE.name,
            "source_sha256": file_sha256(DATA_SOURCE),
            "label_key": LABEL_KEY,
            "seed": SEED,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "min_stratum_count": MIN_STRATUM_COUNT,
        },
        overwrite=SAVE_OVERWRITE,
    )
    print(f"Split saved: {split_path}")


if __name__ == "__main__":
    main()
