# MLFlow helpers: where runs are stored, how configs become params, and which code version produces the run.

import os
import subprocess
from collections.abc import Mapping
from typing import Any

import mlflow

from src.common.path import get_project_root


def local_tracking_uri() -> str:
    """SQLite database at the project root, the default local tracking store."""
    return f"sqlite:///{get_project_root() / 'mlflow.db'}"


def resolve_tracking_uri() -> str:
    """
    MLFLOW_TRACKING_URI wins when set (e.g. a shared tracking server later on).
    Otherwise use the local store at the project root: MLflow's own default is relative to the
    current working directory, so runs started from different directories would be scattered
    across several unrelated databases.
    """
    return os.environ.get("MLFLOW_TRACKING_URI") or local_tracking_uri()


def setup_experiment(name: str) -> str:
    """
    Point MLFlow at the tracking store and make 'name' the active experiment, create it if needed.

    For the local default store, artifacts (checkpoints, reports) go to <project root>/mlruns/<name>;
    MLFlow would otherwise also place them relative to the current working directory.
    """
    tracking_uri = resolve_tracking_uri()
    # Compare URIs instead of checking the environment variable: mlflow.set_tracking_uri writes
    # MLFLOW_TRACKING_URI into os.environ itself
    is_local_store = tracking_uri == local_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    elif is_local_store:
        artifact_location = (get_project_root() / "mlruns" / name).as_uri()
        experiment_id = mlflow.create_experiment(name, artifact_location=artifact_location)
    else:
        experiment_id = mlflow.create_experiment(name)
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def flatten_params(params: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Nested config -> flat MLflow params, e.g. {"pos_weight": {"mode": "sqrt"}} -> {"pos_weight.mode": "sqrt"}."""
    flat: dict[str, str] = {}
    for key, value in params.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, name))
        else:
            flat[name] = str(value)
    return flat


def git_tags() -> dict[str, str]:
    """
    Code version of a run: the commit, and whether the working tree had uncommitted changes.
    A run made with uncommitted changes cannot be reproduced from its commit alone, so that flag
    is worth recording. Returns an empty dict outside a git repository.
    """
    root = get_project_root()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {"git_commit": commit, "git_dirty": str(bool(status.strip())).lower()}
