"""
Turning Optuna trials into configs, and reporting a study, without training anything.
"""

import optuna
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution
from pydantic import ValidationError

from src.common.config_schema import ModelConfig, SearchConfig, TrainConfig, load_config
from src.training.search import (
    count_trials,
    export_trial,
    plot_trials,
    resume_seed,
    slim_model_name,
    suggest_configs,
    to_markdown,
    trial_rows,
    trial_to_configs,
)

BASE = TrainConfig(
    experiment_name="main", split="toy_v1", model="resnet18", epochs=30, batch_size=8, learning_rate=1e-3, augment=True, early_stopping_patience=None, checkpoint="last"
)
SEARCH = SearchConfig.model_validate(
    {
        "study_name": "toy",
        "experiment_name": "toy_search",
        "base_experiment": "toy",
        "epochs": 5,
        "n_trials": 3,
        "seed": 7,
        "space": {
            "width": [8, 16],
            "depth": [1, 2],
            "small_input_stem": [True, False],
            "learning_rate": {"low": 1e-4, "high": 1e-2},
            "weight_decay": {"low": 1e-5, "high": 1e-3},
            "warmup_epochs": [0, 1, 5],
        },
    }
)
PARAMS = {"width": 16, "depth": 1, "small_input_stem": False, "learning_rate": 2e-3, "weight_decay": 1e-4, "warmup_epochs": 1}


def test_suggested_configs_combine_the_trial_and_the_base_experiment():
    train_cfg, model_cfg = suggest_configs(optuna.trial.FixedTrial(PARAMS), SEARCH, BASE)
    assert (model_cfg.name, model_cfg.width, model_cfg.depths, model_cfg.small_input_stem) == ("slim_resnet", 16, [1, 1, 1, 1], False)
    assert (train_cfg.learning_rate, train_cfg.weight_decay, train_cfg.warmup_epochs) == (2e-3, 1e-4, 1)
    # From the search config
    assert (train_cfg.epochs, train_cfg.seed, train_cfg.experiment_name) == (5, 7, "toy_search")
    # Kept from the base experiment
    assert train_cfg.augment and train_cfg.split == "toy_v1" and train_cfg.batch_size == 8
    assert train_cfg.checkpoint == "last" and train_cfg.early_stopping_patience is None


def test_inconsistent_trials_are_rejected_before_training():
    with pytest.raises(ValidationError, match="smaller than epochs"):
        suggest_configs(optuna.trial.FixedTrial({**PARAMS, "warmup_epochs": 5}), SEARCH, BASE)


def make_study() -> optuna.Study:
    """A study with three completed trials; the third is worse than the first on both objectives."""
    distributions = {
        "width": CategoricalDistribution([8, 16]),
        "depth": CategoricalDistribution([1, 2]),
        "small_input_stem": CategoricalDistribution([True, False]),
        "learning_rate": FloatDistribution(1e-4, 1e-2, log=True),
        "weight_decay": FloatDistribution(1e-5, 1e-3, log=True),
        "warmup_epochs": CategoricalDistribution([0, 1, 5]),
    }
    study = optuna.create_study(directions=["minimize", "minimize"])
    for values, width in (((50.0, 10.0), 8), ((42.0, 90.0), 16), ((55.0, 20.0), 16)):
        study.add_trial(optuna.trial.create_trial(params={**PARAMS, "width": width}, distributions=distributions, values=list(values), user_attrs={"run_id": f"run{width}"}))
    return study


def test_trial_rows_flag_the_pareto_front():
    rows = trial_rows(make_study())
    assert [row["mflops"] for row in rows] == [10.0, 20.0, 90.0]
    assert [row["pareto"] for row in rows] == [True, False, True]


def test_report_and_plot(tmp_path):
    rows = trial_rows(make_study())
    table = to_markdown(rows, SEARCH)
    assert sum("| ★ |" in line for line in table.splitlines()) == 2  # the explanation above the table mentions ★ too
    plot_trials(rows, tmp_path / "trials.png", title="toy")
    assert (tmp_path / "trials.png").stat().st_size > 0


def first_params_after_restart(storage: str, seed: int) -> dict:
    """Run 2 trials, then 'restart' the search with a new sampler seeded with `seed`; return the next trial's params."""

    def objective(trial: optuna.Trial) -> float:
        return trial.suggest_float("x", 0.0, 1.0)

    optuna.create_study(study_name="s", storage=storage, sampler=optuna.samplers.TPESampler(seed=0)).optimize(objective, n_trials=2)
    resumed = optuna.create_study(study_name="s", storage=storage, sampler=optuna.samplers.TPESampler(seed=seed), load_if_exists=True)
    resumed.optimize(objective, n_trials=1)
    return resumed.trials[-1].params


def test_resuming_with_the_original_seed_replays_old_trials(tmp_path):
    # The problem resume_seed solves: Optuna does not store the sampler state
    storage = f"sqlite:///{tmp_path / 'replay.db'}"
    params = first_params_after_restart(storage, seed=0)
    assert params == optuna.load_study(study_name="s", storage=storage).trials[0].params


def test_resume_seed_avoids_replaying_old_trials(tmp_path):
    storage = f"sqlite:///{tmp_path / 'resume.db'}"
    assert count_trials("s", storage) == 0
    params = first_params_after_restart(storage, seed=resume_seed(0, existing_trials=2))
    earlier = [trial.params for trial in optuna.load_study(study_name="s", storage=storage).trials[:2]]
    assert params not in earlier
    assert count_trials("s", storage) == 3


# ========== exporting trials ==========


def completed_trial(params: dict, values: tuple[float, float] = (57.3, 192.2)) -> optuna.trial.FrozenTrial:
    distributions = {
        "width": CategoricalDistribution([8, 16, 32]),
        "depth": CategoricalDistribution([1, 2]),
        "small_input_stem": CategoricalDistribution([True, False]),
        "learning_rate": FloatDistribution(1e-4, 1e-2, log=True),
        "weight_decay": FloatDistribution(1e-5, 1e-3, log=True),
        "warmup_epochs": CategoricalDistribution([0, 1, 5]),
    }
    study = optuna.create_study(directions=["minimize", "minimize"])
    for _ in range(3):  # so the exported trial gets a number other than 0
        study.add_trial(optuna.trial.create_trial(params=params, distributions=distributions, values=list(values), user_attrs={"run_id": "abc123"}))
    return study.trials[-1]


def test_slim_model_name():
    assert slim_model_name(ModelConfig(name="slim_resnet", width=16, depths=[2, 2, 2, 2], small_input_stem=True)) == "slim_resnet_w16_d2_small"
    assert slim_model_name(ModelConfig(name="slim_resnet", width=32, depths=[1, 2, 1, 1], small_input_stem=False)) == "slim_resnet_w32_d1-2-1-1_std"


def test_confirmation_configs_use_the_full_schedule_of_the_base_experiment():
    train_cfg, model_cfg = trial_to_configs(PARAMS, SEARCH, BASE)
    # Searched settings come from the trial...
    assert (train_cfg.learning_rate, train_cfg.weight_decay, train_cfg.warmup_epochs, model_cfg.width) == (2e-3, 1e-4, 1, 16)
    # ...but schedule, seed and MLflow experiment from the base experiment, unlike the screening run
    assert (train_cfg.epochs, train_cfg.seed, train_cfg.experiment_name) == (BASE.epochs, BASE.seed, BASE.experiment_name)
    assert train_cfg.model == "slim_resnet_w16_d1_std"


def test_exported_configs_reload_to_the_same_configs(tmp_path):
    params = {**PARAMS, "weight_decay": 1.02e-05}  # a value YAML must keep as a float
    model_path, experiment_path = export_trial(completed_trial(params), SEARCH, BASE, conf_dir=tmp_path)
    assert experiment_path.name == "wafer_toy_t002.yaml"
    expected_train, expected_model = trial_to_configs(params, SEARCH, BASE)
    assert load_config(experiment_path, TrainConfig) == expected_train
    assert load_config(model_path, ModelConfig) == expected_model
    header = experiment_path.read_text(encoding="utf-8")
    assert "trial 2" in header and "abc123" in header and "57.3 validation errors" in header


def test_export_never_silently_replaces_a_different_file(tmp_path):
    trial = completed_trial(PARAMS)
    export_trial(trial, SEARCH, BASE, conf_dir=tmp_path)
    export_trial(trial, SEARCH, BASE, conf_dir=tmp_path)  # identical content: fine
    changed = completed_trial({**PARAMS, "learning_rate": 5e-3})
    with pytest.raises(FileExistsError, match="different content"):
        export_trial(changed, SEARCH, BASE, conf_dir=tmp_path)
    export_trial(changed, SEARCH, BASE, conf_dir=tmp_path, overwrite=True)
