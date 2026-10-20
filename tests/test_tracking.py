import subprocess
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from src.training import tracking
from src.training.tracking import committed_version, flatten_params, git_tags, resolve_tracking_uri, setup_experiment

# ================= fixture: synthesised project root =================


@pytest.fixture
def project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(tracking, "get_project_root", lambda: tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    return tmp_path


# ================= Test Cases =================


def test_flatten_params_nests_with_dots():
    flat = flatten_params({"lr": 0.001, "pos_weight": {"mode": "sqrt", "max_value": None}})
    assert flat == {"lr": "0.001", "pos_weight.mode": "sqrt", "pos_weight.max_value": "None"}


def test_flatten_params_prefix():
    assert flatten_params({"name": "resnet18"}, prefix="model_cfg") == {"model_cfg.name": "resnet18"}


def test_tracking_uri_defaults_to_project_root(project_root):
    assert resolve_tracking_uri() == f"sqlite:///{project_root / 'mlflow.db'}"


def test_tracking_uri_env_override(project_root, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking-server:5000")
    assert resolve_tracking_uri() == "http://tracking-server:5000"


def test_setup_experiment_is_idempotent_and_anchored(project_root):
    first = setup_experiment("unit_test")
    second = setup_experiment("unit_test")
    assert first == second
    assert (project_root / "mlflow.db").exists()
    assert mlflow.get_experiment(first).artifact_location == (project_root / "mlruns" / "unit_test").as_uri()

    # A second experiment in the same process: mlflow.set_tracking_uri has already written
    # MLFLOW_TRACKING_URI into the environment by now, which must not change where artifacts go
    other = setup_experiment("another_experiment")
    assert mlflow.get_experiment(other).artifact_location == (project_root / "mlruns" / "another_experiment").as_uri()


def test_run_logging_as_used_by_the_training_script(project_root):
    setup_experiment("unit_test")
    records = [{"train_loss": 0.9, "val_macro_ap": 0.5}, {"train_loss": 0.4, "val_macro_ap": 0.7}]

    with mlflow.start_run(run_name="baseline") as run:
        mlflow.log_params(flatten_params({"pos_weight": {"mode": "none"}}))
        callback = mlflow.log_metrics  # the trainer calls it as callback(record, epoch)
        for epoch, record in enumerate(records, start=1):
            callback(record, epoch)
        mlflow.log_dict({"Center": 1.39}, "config/pos_weight.json")

    client = MlflowClient()
    history = client.get_metric_history(run.info.run_id, "val_macro_ap")
    assert [(m.step, m.value) for m in history] == [(1, 0.5), (2, 0.7)]
    assert client.get_run(run.info.run_id).data.params["pos_weight.mode"] == "none"
    artifact_file = project_root / "mlruns" / "unit_test" / run.info.run_id / "artifacts" / "config" / "pos_weight.json"
    assert artifact_file.exists()


def test_git_tags_outside_repo_is_empty(project_root):
    assert git_tags() == {}


def test_git_tags_reports_commit_and_dirty_state(project_root):
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=project_root, check=True, capture_output=True)

    git("init")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    (project_root / "a.txt").write_text("a")
    git("add", "a.txt")
    git("commit", "-m", "init")

    tags = git_tags()
    assert len(tags["git_commit"]) == 40
    assert tags["git_dirty"] == "false"

    (project_root / "a.txt").write_text("changed")
    assert git_tags()["git_dirty"] == "true"


# ================= committed_version =================


def test_committed_version_refuses_uncommitted_changes():
    assert committed_version({"git_commit": "abc123", "git_dirty": "false"}) == "abc123"
    assert committed_version({}).startswith("unknown")  # outside a git repository
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        committed_version({"git_commit": "abc123", "git_dirty": "true"})
