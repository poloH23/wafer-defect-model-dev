"""
Create a fixed train/val/test split from a split config.

All dataset- and split-specific values live in conf/.
Add '--overwrite' for overwritting the existing split version.

Usage:
    python scripts/run_split.py --config conf/split/wafer_defect_v1.yaml
"""

import argparse
from pathlib import Path

import numpy as np

from src.common.config_schema import SplitConfig, load_config, load_dataset_config
from src.common.hashing import file_sha256
from src.data.cleaning import find_duplicate_images
from src.data.split import save_split, split_indices, split_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed, stratified train/val/test split.")
    parser.add_argument("--config", required=True, help="Split config path, relative to the project root.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing split with the same version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_cfg = load_config(args.config, SplitConfig)
    dataset_cfg = load_dataset_config(split_cfg.dataset)

    # Images are only loaded when they are needed to find duplicates
    with np.load(dataset_cfg.source_path) as data:
        labels = data[dataset_cfg.label_key]
        images = data[dataset_cfg.image_key] if split_cfg.duplicate else None

    exclude = None
    if images is not None:
        exclude = find_duplicate_images(images, labels)
        print(f"Excluding {exclude.size} duplicate images (the first copy of each is kept)")
        combos, counts = np.unique(labels[exclude], axis=0, return_counts=True)
        for combo, count in sorted(zip(combos, counts, strict=True), key=lambda pair: -pair[1])[:5]:
            name = "+".join(n for n, flag in zip(dataset_cfg.class_names, combo, strict=True) if flag) or "Normal"
            print(f"  {name:<35} {count}")

    result = split_indices(
        labels, val_ratio=split_cfg.val_ratio, test_ratio=split_cfg.test_ratio, seed=split_cfg.seed, min_stratum_count=split_cfg.min_stratum_count, exclude=exclude
    )
    print(f"Split sizes: {result.sizes()}")

    # Stratification check: how far each stratum's val/test share drifts from the target ratio
    rows = split_report(result)
    max_val_dev = max(abs(r["val"] / r["total"] - split_cfg.val_ratio) for r in rows)
    max_test_dev = max(abs(r["test"] / r["total"] - split_cfg.test_ratio) for r in rows)
    smallest = min(rows, key=lambda r: r["total"])
    print(f"Max pre-stratum deviation -> val: {max_val_dev:.3%}, test: {max_test_dev:.3%}")
    print(f"Smallest stratum -> total: {smallest['total']}, train: {smallest['train']}, val: {smallest['val']}, test: {smallest['test']}")

    split_path = save_split(
        result,
        split_cfg.output_dir,
        overwrite=args.overwrite,
        metadata={
            "config_file": Path(args.config).as_posix(),
            "source_file": dataset_cfg.source.as_posix(),
            "source_sha256": file_sha256(dataset_cfg.source_path),
            "label_key": dataset_cfg.label_key,
            **split_cfg.model_dump(mode="json"),  # every split parameter is recored by dumping the whole config
        },
    )
    print(f"Split saved: {split_path}")


if __name__ == "__main__":
    main()
