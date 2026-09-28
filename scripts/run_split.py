"""
Create a fixed train/val/test split from a split config.

All dataset- and split-specific values live in conf/.
Add '--overwrite' for overwritting the existing split version.

Usage:
    python scripts/run_split.py --config conf/split/wafer_defect_v1.yaml
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from src.common.config_schema import SplitConfig, load_config, load_dataset_config
from src.data.split import save_split, split_indices, split_report


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Fingerprint of the source file, so the split can be tied to the exact data it was made from."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed, stratified train/val/test split.")
    parser.add_argument("--config", required=True, help="Split config path, relative to the project root.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing split with the same version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_cfg = load_config(args.config, SplitConfig)
    dataset_cfg = load_dataset_config(split_cfg.dataset)

    # Only the label array of the NPZ data is loaded here, the images are never loaded.
    with np.load(dataset_cfg.source_path) as data:
        labels = data[dataset_cfg.label_key]

    result = split_indices(labels, val_ratio=split_cfg.val_ratio, test_ratio=split_cfg.test_ratio, seed=split_cfg.seed, min_stratum_count=split_cfg.min_stratum_count)
    print(f"Split sizes: {result.sizes()}")

    # Stratification check: how far each stratum's val/test share drifts from the target ratio
    rows = split_report(result)
    max_val_dev = max(abs(r["val"] / r["total"] - split_cfg.val_ratio) for r in rows)
    max_test_dev = max(abs(r["test"] / r["total"] - split_cfg.test_ratio) for r in rows)
    smallest = min(rows, key=lambda r: r["total"])
    print(f"Max pre-stratum deviation -> val: {max_val_dev:.3%}, test: {max_test_dev:.3%}")
    print(f"Smallest stratum -> total: {smallest['total']}, train: {smallest['train']}, val: {smallest['val']}, test: {smallest['test']}")

    split_path = save_split(
        result=result,
        output_dir=split_cfg.output_dir,
        overwrite=args.overwrite,
        metadata={
            "config_file": Path(args.config).as_posix(),
            "source_file": dataset_cfg.source.as_posix(),
            "source_sha256": file_sha256(dataset_cfg.source_path),
            "label_key": dataset_cfg.label_key,
            **split_cfg.model_dump(mode="json"),
        },
    )
    print(f"Split saved: {split_path}")


if __name__ == "__main__":
    main()
