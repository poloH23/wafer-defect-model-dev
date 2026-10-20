"""
Everything an analysis script needs to work on a finished training run: the configs the run
actually used, and the cleaned data, with a check that the split on disk is the one it used.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from src.common.config_schema import DatasetConfig, SplitConfig, TrainConfig, load_config, load_dataset_config, load_split_config
from src.data.cleaning import apply_invalid_pixel_policy
from src.data.io import load_npz_arrays
from src.data.split import load_split_meta
from src.training.tracking import EFFECTIVE_TRAIN_CONFIG, resolve_tracking_uri


@dataclass(frozen=True)
class RunContext:
    run_id: str
    tags: dict[str, str]
    train_cfg: TrainConfig
    split_cfg: SplitConfig
    dataset_cfg: DatasetConfig
    # Number of discrete pixel categories, checked to be set
    num_values: int


def load_train_config(run_id: str, tags: dict[str, str], download_dir: str | Path) -> TrainConfig:
    """
    The training config a run actually used: config/train.json, logged by every run since search
    runs existed (it includes command-line overrides and search-generated settings). Older runs only
    logged their experiment YAML, named by the config_file tag.
    """
    logged = {artifact.path for artifact in MlflowClient().list_artifacts(run_id, "config")}
    if EFFECTIVE_TRAIN_CONFIG in logged:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=EFFECTIVE_TRAIN_CONFIG, dst_path=str(download_dir))
        return TrainConfig.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
    config_name = Path(tags["config_file"]).name
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=f"config/{config_name}", dst_path=str(download_dir))
    return load_config(path, TrainConfig)


def load_run_context(run_id: str, download_dir: str | Path) -> RunContext:
    """
    Load the configs of a finished run.

    The training config is the copy logged with the run, not the current YAML file, which may have
    changed since. The split must have the same digest as the one recorded by the run.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri())
    run = MlflowClient().get_run(run_id)
    train_cfg = load_train_config(run_id, dict(run.data.tags), download_dir)
    split_cfg = load_split_config(train_cfg.split)
    dataset_cfg = load_dataset_config(split_cfg.dataset)
    if dataset_cfg.num_pixel_values is None:
        raise ValueError("This analysis expects discrete-valued maps; set num_pixel_values in the dataset config.")
    if load_split_meta(split_cfg.output_dir)["split_digest"] != run.data.params["split_digest"]:
        raise RuntimeError(f"Split '{train_cfg.split}' on disk differs from the one run {run_id} used")

    return RunContext(
        run_id=run_id,
        tags=dict(run.data.tags),
        train_cfg=train_cfg,
        split_cfg=split_cfg,
        dataset_cfg=dataset_cfg,
        num_values=dataset_cfg.num_pixel_values,
    )


def find_checkpoint_artifact(ctx: RunContext) -> str:
    """
    Path of the run's saved model: the checkpoint_artifact tag when the run has one, otherwise the
    single .pt file logged under checkpoints/ (runs saved before the tag existed, or without it).
    """
    if "checkpoint_artifact" in ctx.tags:
        return ctx.tags["checkpoint_artifact"]
    candidates = [a.path for a in MlflowClient().list_artifacts(ctx.run_id, "checkpoints") if a.path.endswith(".pt")]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one .pt file under checkpoints/ in run {ctx.run_id}, found {candidates}")
    return candidates[0]


def download_checkpoint(ctx: RunContext, download_dir: str | Path) -> str:
    """Download the run's saved model and return its local path."""
    return mlflow.artifacts.download_artifacts(run_id=ctx.run_id, artifact_path=find_checkpoint_artifact(ctx), dst_path=str(download_dir))


def load_clean_arrays(ctx: RunContext) -> tuple[np.ndarray, np.ndarray]:
    """Images and labels with the same pixel cleaning the run was trained with."""
    images, labels = load_npz_arrays(ctx.dataset_cfg.source_path, ctx.dataset_cfg.image_key, ctx.dataset_cfg.label_key)
    images, _ = apply_invalid_pixel_policy(images, ctx.num_values, ctx.dataset_cfg.invalid_pixel_policy)
    return images, labels
