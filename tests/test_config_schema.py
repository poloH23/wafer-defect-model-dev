from pathlib import Path

import pytest
from pydantic import ValidationError

from src.common.config_schema import (
    DatasetConfig,
    EdaConfig,
    ModelConfig,
    SearchConfig,
    SplitConfig,
    TestEvaluationConfig,
    TrainConfig,
    load_config,
    load_named_config,
)
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


def test_valid_dataset_config(tmp_path):
    cfg = load_config(write(tmp_path, DATASET_YAML), DatasetConfig)
    assert cfg.class_name_map == {"class_0": "a", "class_1": "b", "class_2": "c"}
    assert cfg.source_path.is_absolute()


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
    "search": SearchConfig,
}


def committed_configs() -> list[tuple[str, str]]:
    conf_dir = get_project_root() / "conf"
    return [(group, path.stem) for group in GROUP_SCHEMAS for path in sorted((conf_dir / group).glob("*.yaml"))]


@pytest.mark.parametrize(("group", "name"), committed_configs())
def test_committed_config_is_valid(group, name):
    cfg = load_named_config(group, name, GROUP_SCHEMAS[group])
    if isinstance(cfg, TrainConfig):
        # Cross-references must resolve too
        load_named_config("split", cfg.split, SplitConfig)
        load_named_config("model", cfg.model, ModelConfig)
    if isinstance(cfg, SearchConfig):
        load_named_config("experiments", cfg.base_experiment, TrainConfig)


TRAIN_YAML = """
experiment_name: toy
split: toy_v1
model: resnet18
epochs: 10
batch_size: 8
learning_rate: 1.0e-3
checkpoint: last
"""


def test_last_checkpoint_without_early_stopping_is_valid(tmp_path):
    cfg = load_config(write(tmp_path, TRAIN_YAML), TrainConfig)
    assert cfg.checkpoint == "last"
    assert cfg.early_stopping_patience is None


def test_last_checkpoint_rejects_early_stopping(tmp_path):
    with pytest.raises(ValidationError, match="cannot be combined with early stopping"):
        load_config(write(tmp_path, TRAIN_YAML + "early_stopping_patience: 5\n"), TrainConfig)


# ========== slim_resnet, warmup and search configs ==========


def test_width_and_depths_only_apply_to_slim_resnet():
    assert ModelConfig(name="slim_resnet", width=16, depths=[1, 1, 1, 1]).width == 16
    with pytest.raises(ValidationError, match="only apply to slim_resnet"):
        ModelConfig(name="resnet18", width=16)


def test_slim_resnet_rejects_empty_stages_and_pretrained_weights():
    with pytest.raises(ValidationError, match="at least one block"):
        ModelConfig(name="slim_resnet", depths=[1, 0, 1, 1])
    with pytest.raises(ValidationError, match="no pretrained weights"):
        ModelConfig(name="slim_resnet", pretrained=True)


def test_warmup_must_be_shorter_than_training(tmp_path):
    assert load_config(write(tmp_path, TRAIN_YAML + "warmup_epochs: 3\n"), TrainConfig).warmup_epochs == 3
    with pytest.raises(ValidationError, match="smaller than epochs"):
        load_config(write(tmp_path, TRAIN_YAML + "warmup_epochs: 10\n"), TrainConfig)


SEARCH_YAML = """
study_name: toy_search
experiment_name: toy_search
base_experiment: toy
epochs: 5
n_trials: 4
space:
  width: [8, 16]
  depth: [1]
  small_input_stem: [true]
  learning_rate: {low: 1.0e-4, high: 1.0e-2}
  weight_decay: {low: 1.0e-5, high: 1.0e-3}
  warmup_epochs: [0, 1]
"""


def test_search_config(tmp_path):
    cfg = load_config(write(tmp_path, SEARCH_YAML), SearchConfig)
    assert cfg.last_k == 3 and cfg.space.learning_rate.log


def test_search_range_must_be_ordered(tmp_path):
    with pytest.raises(ValidationError, match="low must be smaller than high"):
        load_config(write(tmp_path, SEARCH_YAML.replace("low: 1.0e-4, high: 1.0e-2", "low: 1.0e-2, high: 1.0e-4")), SearchConfig)


def test_test_evaluation_config_rejects_inconsistencies():
    selected = {"experiment": "a", "overrides": "none"}
    assert TestEvaluationConfig(selected=selected).deliverable_seed == 0
    with pytest.raises(ValidationError, match="deliverable_seed"):
        TestEvaluationConfig(selected=selected, seeds=[1, 2], deliverable_seed=0)
    with pytest.raises(ValidationError, match="appear once"):
        TestEvaluationConfig(selected=selected, others=[selected])
