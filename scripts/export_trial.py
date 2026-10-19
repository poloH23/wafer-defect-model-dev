"""
Turn completed trials of a search into model and experiment configs for their confirmation runs
(3 seeds, full schedule; see docs/model_selection.md).

Usage:
    python scripts/export_trial.py --config conf/search/slim_resnet_v1.yaml --trials 15 46 10 29 36
    make train-seeds TRAIN_CONFIG=conf/experiments/wafer_slim_resnet_v1_t029.yaml
"""

import argparse

import optuna

from src.common.config_schema import SearchConfig, load_config, load_experiment_config
from src.common.path import get_project_root
from src.training.search import export_trial

OPTUNA_DB = "optuna.db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export search trials as configs for confirmation runs.")
    parser.add_argument("--config", required=True, help="Search config path, e.g. conf/search/slim_resnet_v1.yaml.")
    parser.add_argument("--trials", type=int, nargs="+", required=True, help="Trial numbers to export.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing config files whose content differs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_cfg = load_config(args.config, SearchConfig)
    base_cfg = load_experiment_config(search_cfg.base_experiment)
    study = optuna.load_study(study_name=search_cfg.study_name, storage=f"sqlite:///{(get_project_root() / OPTUNA_DB).as_posix()}")
    trials = {trial.number: trial for trial in study.trials}

    for number in args.trials:
        if number not in trials:
            raise ValueError(f"Study '{search_cfg.study_name}' has not trial {number}")
        model_path, experiment_path = export_trial(trials[number], search_cfg, base_cfg, conf_dir=get_project_root() / "conf", overwrite=args.overwrite)
        print(f"Trial {number}: {experiment_path.relative_to(get_project_root())} (model {model_path.relative_to(get_project_root())})")


if __name__ == "__main__":
    main()
