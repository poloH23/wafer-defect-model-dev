"""
Multi-objective search over slim ResNets with Optuna: minimize validation errors and FLOPs.

Usage:
    python scripts/search.py --config conf/search/slim_resnet_v1.yaml

Each trial is a full training run recorded in MLflow (experiment from the search config), on the
validation split only. The study itself is stored in optuna.db at the project root, so an
interrupted search resumes where it stopped: running the same command again completes the missing
trials. Optuna cannot prune trials of a multi-objective study, so every trial trains for the full
number of epochs of the search config.

The search screens candidates with a single seed and a short schedule. Its results are not
conclusions: Pareto-optimal configurations are retrained with 3 seeds and the full schedule, then
compared and selected as described in docs/model_selection.md.
"""

import argparse
from pathlib import Path

import optuna
from optuna.trial import TrialState

from src.common.config_schema import SearchConfig, load_config, load_experiment_config, named_config_path, resolve_path
from src.common.path import get_project_root
from src.training.pipeline import prepare_data, train_and_log
from src.training.search import count_trials, plot_trials, resume_seed, suggest_configs, to_markdown, trial_rows
from src.training.trainer import resolve_device

OPTUNA_DB = "optuna.db"
REPORT_DIR = Path("reports/experiments")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-objective search (validation errors vs FLOPs) over slim ResNets.")
    parser.add_argument("--config", required=True, help="Search config path, e.g. conf/search/slim_resnet_v1.yaml.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_cfg = load_config(args.config, SearchConfig)
    base_cfg = load_experiment_config(search_cfg.base_experiment)
    config_files = [resolve_path(args.config), named_config_path("experiments", search_cfg.base_experiment)]
    device = resolve_device()
    data = prepare_data(base_cfg.split)

    storage = f"sqlite:///{(get_project_root() / OPTUNA_DB).as_posix()}"
    # See resume_seed: a resumed search must not replay the suggestions of the original one
    sampler_seed = resume_seed(search_cfg.seed, count_trials(search_cfg.study_name, storage))
    study = optuna.create_study(
        study_name=search_cfg.study_name,
        storage=storage,
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        train_cfg, model_cfg = suggest_configs(trial, search_cfg, base_cfg)
        outcome = train_and_log(
            train_cfg,
            model_cfg,
            data,
            device,
            run_name=f"{search_cfg.study_name}_trial{trial.number:03d}",
            tags={"config_file": Path(args.config).as_posix(), "study": search_cfg.study_name, "trial": str(trial.number), "overrides": "none"},
            config_files=config_files,
        )
        trial.set_user_attr("run_id", outcome.run_id)
        trial.set_user_attr("num_parameters", outcome.num_parameters)
        return outcome.mean_val_errors(search_cfg.last_k), outcome.flops / 1e6

    completed = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
    remaining = search_cfg.n_trials - completed
    print(f"Study '{search_cfg.study_name}': {completed} trials completed, {max(remaining, 0)} to go (sampler seed {sampler_seed})")
    if remaining > 0:
        # An exception stops the search instead of silently failing trial after trial; fix it and rerun to resume
        study.optimize(objective, n_trials=remaining)

    rows = trial_rows(study)
    report_dir = resolve_path(REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)
    table = to_markdown(rows, search_cfg)
    (report_dir / f"{search_cfg.study_name}.md").write_text(table, encoding="utf-8")
    plot_trials(rows, report_dir / f"{search_cfg.study_name}.png", title=f"{search_cfg.study_name}: validation errors vs FLOPs")
    print(table)
    print(f"Saved: {(REPORT_DIR / search_cfg.study_name).as_posix()}.md and .png")


if __name__ == "__main__":
    main()
