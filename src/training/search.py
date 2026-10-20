"""
The parts of the multi-objective search (scripts/search.py) that need no training: turning an Optuna
trial into validated configs, and reporting the trade-off between validation errors and FLOPs.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import optuna
import yaml
from matplotlib.ticker import FuncFormatter
from optuna.trial import FrozenTrial, TrialState

from src.common.config_schema import CONF_DIR, ModelConfig, SearchConfig, TrainConfig

# Both minimized: misclassified validation samples, and compute per image
OBJECTIVES = ("val_errors", "mflops")


def count_trials(study_name: str, storage: str) -> int:
    """Trials already stored for a study, in any state; 0 if the study does not exist yet."""
    try:
        return len(optuna.load_study(study_name=study_name, storage=storage).trials)
    except KeyError:
        return 0


def resume_seed(base_seed: int, existing_trials: int) -> int:
    """
    Sampler seed for a search that starts or resumes with `existing_trials` trials in its study.

    Optuna does not store the sampler's random state. A resumed search seeded like the original
    would replay the original's first suggestions, retraining configurations already evaluated:
    the first trials of TPE are random, so every interruption would waste them again. Offsetting the
    seed by the number of stored trials keeps every (re)start deterministic without repeating them.
    """
    return base_seed + existing_trials


def suggest_configs(trial: optuna.Trial, search_cfg: SearchConfig, base_cfg: TrainConfig) -> tuple[TrainConfig, ModelConfig]:
    """
    One point of the search space as a training config and a slim_resnet model config. Settings not
    searched come from the base experiment. The same depth is used in all four stages, which keeps
    the space small enough for a few dozen trials.
    """
    space = search_cfg.space
    width = trial.suggest_categorical("width", space.width)
    depth = trial.suggest_categorical("depth", space.depth)
    small_input_stem = trial.suggest_categorical("small_input_stem", space.small_input_stem)
    learning_rate = trial.suggest_float("learning_rate", space.learning_rate.low, space.learning_rate.high, log=space.learning_rate.log)
    weight_decay = trial.suggest_float("weight_decay", space.weight_decay.low, space.weight_decay.high, log=space.weight_decay.log)
    warmup_epochs = trial.suggest_categorical("warmup_epochs", space.warmup_epochs)

    model_cfg = ModelConfig(name="slim_resnet", width=width, depths=[depth] * 4, small_input_stem=small_input_stem)
    # Validated from scratch (model_copy would skip validation), so that an inconsistent combination,
    # such as a warmup as long as the search's epochs, fails here instead of in the middle of training
    train_cfg = TrainConfig.model_validate(
        {
            **base_cfg.model_dump(),
            "experiment_name": search_cfg.experiment_name,
            "model": "slim_resnet",
            "epochs": search_cfg.epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "seed": search_cfg.seed,
            "checkpoint": "last",
            "early_stopping_patience": None,
        }
    )
    return train_cfg, model_cfg


def trial_rows(study: optuna.Study) -> list[dict[str, Any]]:
    """Completed trials, cheapest first, flagged when they are on the Pareto front."""
    pareto = {trial.number for trial in study.best_trials}
    rows = [
        {
            "trial": trial.number,
            "pareto": trial.number in pareto,
            **dict(zip(OBJECTIVES, trial.values, strict=True)),
            **trial.params,
            "run_id": trial.user_attrs.get("run_id"),
        }
        for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    ]
    return sorted(rows, key=lambda row: (row["mflops"], row["val_errors"]))


def to_markdown(rows: list[dict[str, Any]], search_cfg: SearchConfig) -> str:
    lines = [
        f"Search `{search_cfg.study_name}`: {len(rows)} completed trials of slim ResNets, {search_cfg.epochs} epochs each "
        f"(seed {search_cfg.seed}), scored by validation errors averaged over the last {search_cfg.last_k} epochs. "
        "Single-seed screening results: Pareto-optimal trials (★) are retrained with 3 seeds and the full schedule before any conclusion.",
        "",
        "| Trial | Pareto | MFLOPs | Val errors | Width | Depth | Small stem | Learning rate | Weight decay | Warmup | MLflow run |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['trial']} | {'★' if r['pareto'] else ''} | {r['mflops']:,.1f} | {r['val_errors']:.1f} | {r['width']} | {r['depth']} | "
            f"{r['small_input_stem']} | {r['learning_rate']:.2e} | {r['weight_decay']:.2e} | {r['warmup_epochs']} | {r['run_id']} |"
        )
    return "\n".join(lines) + "\n"


def plot_trials(rows: list[dict[str, Any]], path: str | Path, title: str) -> None:
    """Validation errors against FLOPs (log scale), with the Pareto front highlighted and labeled by trial number."""
    figure, ax = plt.subplots(figsize=(9, 5.5))
    others = [r for r in rows if not r["pareto"]]
    front = sorted((r for r in rows if r["pareto"]), key=lambda r: r["mflops"])
    ax.scatter([r["mflops"] for r in others], [r["val_errors"] for r in others], color="tab:gray", alpha=0.6, label="trial")
    ax.plot([r["mflops"] for r in front], [r["val_errors"] for r in front], "o-", color="tab:red", label="Pareto front")
    for r in front:
        ax.annotate(f"#{r['trial']}", (r["mflops"], r["val_errors"]), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xscale("log")
    # Plain numbers (10, 100, 1000) instead of the default scientific notation of log axes
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel("MFLOPs per image (log scale)")
    ax.set_ylabel("Validation errors (mean of last epochs)")
    ax.set_title(title)
    ax.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


# ========== exporting trials for confirmation ==========


def slim_model_name(model_cfg: ModelConfig) -> str:
    """File name of a slim_resnet model config, e.g. slim_resnet_w16_d2_small."""
    depths = model_cfg.depths
    depth = str(depths[0] if len(set(depths)) == 1 else "-".join(map(str, depths)))
    return f"slim_resnet_w{model_cfg.width}_d{depth}_{'small' if model_cfg.small_input_stem else 'std'}"


def trial_to_configs(params: dict[str, Any], search_cfg: SearchConfig, base_cfg: TrainConfig) -> tuple[TrainConfig, ModelConfig]:
    """
    The configs to retrain a trial for confirmation: the trial's architecture and hyperparameters,
    but the base experiment's full schedule, seed and MLflow experiment, so the confirmation runs
    are directly comparable with the other architectures.
    """
    screening_cfg, model_cfg = suggest_configs(optuna.trial.FixedTrial(params), search_cfg, base_cfg)
    train_cfg = TrainConfig.model_validate(
        {
            **screening_cfg.model_dump(),
            "experiment_name": base_cfg.experiment_name,
            "epochs": base_cfg.epochs,
            "seed": base_cfg.seed,
            "model": slim_model_name(model_cfg),
        }
    )
    return train_cfg, model_cfg


def _write_yaml(path: Path, header: list[str], data: dict[str, Any], overwrite: bool) -> None:
    text = "\n".join(f"# {line}".rstrip() for line in header) + "\n\n" + yaml.safe_dump(data, sort_keys=False, default_flow_style=None)
    if path.exists() and path.read_text(encoding="utf-8") != text and not overwrite:
        raise FileExistsError(f"{path} exists with different content; pass overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def export_trial(trial: FrozenTrial, search_cfg: SearchConfig, base_cfg: TrainConfig, conf_dir: Path = CONF_DIR, overwrite: bool = False) -> tuple[Path, Path]:
    """
    Write conf/model/<slim model>.yaml and conf/experiments/wafer_<study>_t<trial>.yaml for one
    completed trial, with its provenance in the header. Generated rather than copied by hand, so the
    confirmation runs train exactly what the search evaluated. An existing file with different
    content is never replaced silently.
    """
    if trial.state != TrialState.COMPLETE:
        raise ValueError(f"Trial {trial.number} is {trial.state.name}, only completed trials can be exported")
    train_cfg, model_cfg = trial_to_configs(trial.params, search_cfg, base_cfg)
    val_errors, mflops = trial.values
    provenance = [
        f"Generated by scripts/export_trial.py from search '{search_cfg.study_name}', trial {trial.number}",
        f"(MLflow run {trial.user_attrs.get('run_id')}). Screening result: {val_errors:.1f} validation errors",
        f"after {search_cfg.epochs} epochs with seed {search_cfg.seed}, {mflops:,.1f} MFLOPs per image.",
    ]

    model_path = conf_dir / "model" / f"{train_cfg.model}.yaml"
    _write_yaml(
        model_path,
        [f"Slim ResNet: width {model_cfg.width}, blocks per stage {model_cfg.depths}, {'small-input' if model_cfg.small_input_stem else 'standard'} stem.", *provenance],
        model_cfg.model_dump(mode="json"),
        overwrite,
    )

    experiment_path = conf_dir / "experiments" / f"wafer_{search_cfg.study_name}_t{trial.number:03d}.yaml"
    _write_yaml(
        experiment_path,
        [
            *provenance,
            "",
            f"Confirmation run: identical to {search_cfg.base_experiment}.yaml except for the model and the searched",
            f"hyperparameters (learning rate, weight decay, warmup), trained for the full {train_cfg.epochs} epochs.",
        ],
        train_cfg.model_dump(mode="json"),
        overwrite,
    )
    return model_path, experiment_path
