"""
Everything an analysis script needs to work on a finished training run: the configs the run
actually used, and the cleaned data, with a check that the split on disk is the one it used.
"""

from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from src.common.config_schema import DatasetConfig, SplitConfig, TrainConfig, load_config, load_dataset_config, load_split_config
from src.data.cleaning import apply_invalid_pixel_policy
from src.data.io import load_npz_arrays
from src.data.split import load_split_meta
from src.training.tracking import resolve_tracking_uri


@dataclass(frozen=True)
class RunContext:
    run_id: str
    train_cfg: TrainConfig
    split_cfg: SplitConfig
    dataset_cfg: DatasetConfig
    num_values: int


def load_run_context(run_id: str, download_dir: str | Path) -> RunContext:
    """
    Load the configs of a finished run.

    The training config is the copy logged with the run, not the current YAML file, which may have
    changed since. The split must have the same digest as the one recorded by the run.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri())
    run = MlflowClient().get_run(run_id)
    config_name = Path(run.data.tags["config_file"]).name
    config_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=f"config/{config_name}", dst_path=str(download_dir))

    train_cfg = load_config(config_path, TrainConfig)
    split_cfg = load_split_config(train_cfg.split)
    dataset_cfg = load_dataset_config(split_cfg.dataset)
    if dataset_cfg.num_pixel_values is None:
        raise ValueError("This analysis expects discrete-valued maps; set num_pixel_values in the dataset config.")
    if load_split_meta(split_cfg.output_dir)["split_digest"] != run.data.params["split_digest"]:
        raise RuntimeError(f"Split '{train_cfg.split}' on disk differs from the one run {run_id} used")
    return RunContext(run_id=run_id, train_cfg=train_cfg, split_cfg=split_cfg, dataset_cfg=dataset_cfg, num_values=dataset_cfg.num_pixel_values)


def load_clean_arrays(ctx: RunContext) -> tuple[np.ndarray, np.ndarray]:
    """Images and labels with the same pixel cleaning the run was trained with."""
    images, labels = load_npz_arrays(ctx.dataset_cfg.source_path, ctx.dataset_cfg.image_key, ctx.dataset_cfg.label_key)
    images, _ = apply_invalid_pixel_policy(images, ctx.num_values, ctx.dataset_cfg.invalid_pixel_policy)
    return images, labels
