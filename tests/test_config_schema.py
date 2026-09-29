from pathlib import Path

import pytest
from pydantic import ValidationError

from src.common.config_schema import DatasetConfig, EdaConfig, ModelConfig, SplitConfig, TrainConfig, load_config, load_named_config
from src.common.path import get_project_root

DATASET_YAML = """
name: toy
source: data/raw/toy.npz
image_key: images
label_key: labels
num_classes: 3
class_names: [a, b, c]
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_class_name_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="expected num_classes"):
        load_config(write(tmp_path, DATASET_YAML.replace("[a, b, c]", "[a, b]")), DatasetConfig)


def test_duplicate_class_name_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="duplicates"):
        load_config(write(tmp_path, DATASET_YAML.replace("[a, b, c]", "[a, b, b]")), DatasetConfig)


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="Extra inputs"):
        load_config(write(tmp_path, DATASET_YAML + "num_clases: 3\n"), DatasetConfig)


def test_split_ratio_sum_is_checked(tmp_path):
    text = "dataset: toy\nversion: v1\nseed: 0\nval_ratio: 0.5\ntest_ratio: 0.5\n"
    with pytest.raises(ValidationError, match="less than 1"):
        load_config(write(tmp_path, text), SplitConfig)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml", DatasetConfig)


# ========== every committed config must be valid ==========

GROUP_SCHEMAS = {
    "dataset": DatasetConfig,
    "split": SplitConfig,
    "eda": EdaConfig,
    "model": ModelConfig,
    "experiments": TrainConfig,
}


def committed_configs() -> list[tuple[str, str]]:
    conf_dir = get_project_root() / "conf"
    return [(group, path.stem) for group in GROUP_SCHEMAS for path in sorted((conf_dir / group).glob("*.yaml"))]


@pytest.mark.parametrize(("group", "name"), committed_configs())
def test_committed_config_is_valid(group, name):
    cfg = load_named_config(group, name, GROUP_SCHEMAS[group])
    if isinstance(cfg, TrainConfig):
        load_named_config("split", cfg.split, SplitConfig)
        load_named_config("model", cfg.model, ModelConfig)
