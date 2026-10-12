import tempfile
from pathlib import Path

import mlflow
import pytest

from src.common.config_schema import DatasetConfig, SplitConfig, TrainConfig
from src.evaluation.run_context import RunContext, find_checkpoint_artifact
from src.training import tracking
from src.training.tracking import setup_experiment


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
