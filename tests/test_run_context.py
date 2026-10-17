import tempfile
from pathlib import Path

import mlflow
import pytest

from src.common.config_schema import DatasetConfig, SplitConfig, TrainConfig
from src.evaluation.run_context import RunContext, find_checkpoint_artifact, load_train_config
from src.training import tracking
from src.training.tracking import EFFECTIVE_TRAIN_CONFIG, setup_experiment


@pytest.fixture(autouse=True)
def tracking_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(tracking, "get_project_root", lambda: tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    setup_experiment("run_context_test")
    return tmp_path


def log_run(files: list[str], tags: dict[str, str] | None = None) -> str:
    with tempfile.TemporaryDirectory() as tmp, mlflow.start_run() as run:
        mlflow.set_tags(tags or {})
        for name in files:
            path = Path(tmp) / name
            path.write_bytes(b"weights")
            mlflow.log_artifact(str(path), artifact_path="checkpoints")
    return run.info.run_id


def context(run_id: str, tags: dict[str, str]) -> RunContext:
    # Only run_id and tags matter for locating the checkpoint
    train_cfg = TrainConfig(experiment_name="e", split="s", model="m", epochs=1, batch_size=1, learning_rate=0.1)
    split_cfg = SplitConfig(dataset="d", version="v1", seed=0, val_ratio=0.1, test_ratio=0.1)
    dataset_cfg = DatasetConfig(name="d", source=Path("x.npz"), image_key="a", label_key="b", num_classes=1, class_names=["c"], num_pixel_values=3)
    return RunContext(run_id=run_id, tags=tags, train_cfg=train_cfg, split_cfg=split_cfg, dataset_cfg=dataset_cfg, num_values=3)


def test_tag_wins_when_present():
    tags = {"checkpoint_artifact": "checkpoints/model.pt"}
    run_id = log_run(["model.pt"], tags)
    assert find_checkpoint_artifact(context(run_id, tags)) == "checkpoints/model.pt"


@pytest.mark.parametrize("filename", ["model.pt", "best_model.pt"])
def test_single_checkpoint_is_found_without_a_tag(filename):
    run_id = log_run([filename])
    assert find_checkpoint_artifact(context(run_id, {})) == f"checkpoints/{filename}"


def test_ambiguous_or_missing_checkpoint_raises():
    with pytest.raises(FileNotFoundError, match="exactly one"):
        find_checkpoint_artifact(context(log_run(["a.pt", "b.pt"]), {}))
    with pytest.raises(FileNotFoundError, match="exactly one"):
        find_checkpoint_artifact(context(log_run([]), {}))


# ========== the training config of a run ==========

TRAIN_CFG = TrainConfig(experiment_name="e", split="s", model="m", epochs=1, batch_size=1, learning_rate=0.1)


def test_effective_config_is_preferred(tmp_path):
    # e.g. a search trial: its config_file is a search config, which is not a training config
    effective = TRAIN_CFG.model_copy(update={"learning_rate": 0.003})
    with mlflow.start_run() as run:
        mlflow.log_dict(effective.model_dump(mode="json"), EFFECTIVE_TRAIN_CONFIG)
        mlflow.log_dict({"study_name": "not a training config"}, "config/search.yaml")
    assert load_train_config(run.info.run_id, {"config_file": "conf/search/search.yaml"}, tmp_path) == effective


def test_older_runs_fall_back_to_their_experiment_yaml(tmp_path):
    yaml_path = tmp_path / "baseline.yaml"
    yaml_path.write_text(
        """
experiment_name: e
split: s
model: m
epochs: 1
batch_size: 1
learning_rate: 1.0e-1
""",
        encoding="utf-8",
    )
    with mlflow.start_run() as run:
        mlflow.log_artifact(str(yaml_path), artifact_path="config")
    assert load_train_config(run.info.run_id, {"config_file": "conf/experiments/baseline.yaml"}, tmp_path / "download") == TRAIN_CFG
