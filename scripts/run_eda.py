"""
Run the dataset analysis pipeline from an EDA config.

All dataset- and split-specific values live in conf/.

Usage:
    python scripts/run_eda.py --config conf/eda/wafer_defect.yaml
"""

import argparse
from typing import Any

from matplotlib.colors import BoundaryNorm, ListedColormap

from src.common.config_schema import EdaConfig, load_config, load_dataset_config
from src.pipelines.eda_pipeline import DatasetAnalysisPipeline


def build_discrete_plot_kwargs(colors: list[str]) -> dict[str, Any]:
    """
    Colormap for images whose pixel values are discrete categories 0..N-1.
    Boundaries [0, 1, 2, ..., N] put each integer value in its own color bin.
    """
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(list(range(len(colors) + 1)), cmap.N)
    return {"cmap": cmap, "norm": norm}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dataset EDA pipeline.")
    parser.add_argument("--config", required=True, help="EDA config path, relative to the porject root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eda_cfg = load_config(args.config, EdaConfig)
    dataset_cfg = load_dataset_config(eda_cfg.dataset)

    plot_kwargs = build_discrete_plot_kwargs(eda_cfg.plot.discrete_colors) if eda_cfg.plot.discrete_colors else {}

    pipeline = DatasetAnalysisPipeline(report_dir=eda_cfg.report_path)
    pipeline.run(
        data_source=dataset_cfg.source_path,
        image_key=dataset_cfg.image_key,
        label_key=dataset_cfg.label_key,
        plot_kwargs=plot_kwargs,
        class_names=dataset_cfg.class_name_map,
        pure_class_examples=eda_cfg.pure_class_examples,
    )


if __name__ == "__main__":
    main()
