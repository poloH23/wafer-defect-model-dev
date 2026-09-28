"""
Save all 8 D4 variants (4 rotation X optional flip) of one sample as a single images grid,
to visually confirm the augmentation before using it in training.

Usage:
    python scripts/preview_augmentation.py --config conf/eda/wafer_defect.yaml --index 0
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.common.config_schema import EdaConfig, load_config, load_dataset_config
from src.data.transforms import apply_dihedral
from src.eda.plotting import build_discrete_plot_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview the 8 D4 augmentation variants of one sample.")
    parser.add_argument("--config", required=True, help="EDA config path, relative to the porject root.")
    parser.add_argument("--index", type=int, default=0, help="Index of the sample to preview.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eda_cfg = load_config(args.config, EdaConfig)
    dataset_cfg = load_dataset_config(eda_cfg.dataset)

    with np.load(dataset_cfg.source_path) as data:
        image = data[dataset_cfg.image_key][args.index]
        label = data[dataset_cfg.label_key][args.index]

    active = [name for name, flag in zip(dataset_cfg.class_name, np.atleast_1d(label), strict=False) if flag]
    label_text = "+".join(active) if active else "Normal"
    plot_kwargs = build_discrete_plot_kwargs(eda_cfg.plot.discrete_colors) if eda_cfg.plot.discrete_colors else {}

    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
    for flip, row in zip((False, True), axes, strict=True):
        for k, ax in enumerate(row):
            ax.imshow(apply_dihedral(image, k, flip), **plot_kwargs)
            ax.set_title(f"rot {k * 90} degrees" + (" + flip" if flip else ""))
            ax.axis("off")
    fig.suptitle(f"D4 augmentation preview | sample #{args.index} | {label_text}")
    fig.tight_layout()

    save_path = eda_cfg.report_path / "augmentation_preview.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Augmentation preview saved: {save_path}")


if __name__ == "__main__":
    main()
