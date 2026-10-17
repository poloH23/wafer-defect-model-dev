"""
Schemas and loaders for task configs stored under conf/.
The schemas are generic (they discribe what any image classification dataset, split or EDA run needs);
the project-specific values lives only in the YAML files.

Layout:
    conf/dataset/<name>.yaml      : facts about the a dataset, shared by every tasks
    conf/split/<name>.yaml        : parameters of one split version
    conf/eda/<name>.yaml          : parameters of one EDA run
    conf/model/<name>.yaml        : model architecture options
    conf/experiments/<name>.yaml  : one training run (references dataset split and model by name)
"""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.common.path import get_project_root

CONF_DIR = Path("conf")


def resolve_path(path: str | Path) -> Path:
    """Paths in configs are relative to the project root, so they work from any working directory."""
    path = Path(path)
    return path if path.is_absolute() else (get_project_root() / path).resolve()


class StrictModel(BaseModel):
    # extra="forbid": an unknown key (usually a typo) is an error, not silently ignored
    # frozen=True: a loaded config cannot be modified halfway through a run
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(StrictModel):
    name: str
    source: Path
    image_key: str
    label_key: str
    num_classes: int = Field(gt=0)  # prevents the missing or extract entry in class_names
    class_names: list[str] = Field(min_length=1)  # order matters
    num_pixel_values: int | None = Field(default=None, ge=2)  # discrete pixel category number (None for continuous images)
    invalid_pixel_policy: Literal["error", "neighbor_mode"] = "error"

    @model_validator(mode="after")
    def check_class_names(self) -> Self:
        if len(self.class_names) != self.num_classes:
            raise ValueError(f"class_names has {len(self.class_names)} entries, expected num_classes = {self.num_classes}")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names contains duplicates")
        return self

    @property
    def source_path(self) -> Path:
        return resolve_path(self.source)

    @property
    def class_name_map(self) -> dict[str, str]:
        """Map the generic ids produced by the analysis modules (class_0, class_1, ...) to readable names."""
        return {f"class_{i}": name for i, name in enumerate(self.class_names)}


class SplitConfig(StrictModel):
    dataset: str
    version: str
    seed: int
    val_ratio: float = Field(gt=0, lt=1)
    test_ratio: float = Field(gt=0, lt=1)
    min_stratum_count: int = Field(default=10, ge=1)
    duplicate: bool = False
    output_root: Path = Path("data/splits")

    @model_validator(mode="after")
    def check_ratio_sum(self) -> Self:
        if self.val_ratio + self.test_ratio >= 1.0:
            raise ValueError("val_ratio + test_ratio must be less than 1.0")
        return self

    @property
    def output_dir(self) -> Path:
        return resolve_path(self.output_root / self.dataset / self.version)


class PlotConfig(StrictModel):
    # For images whose pixel values are discrete categories 0..N-1 (e.g. wafer maps), one color per value:
    # leave empty to use matplotlib's default continuous colormap.
    discrete_colors: list[str] | None = None


class EdaConfig(StrictModel):
    dataset: str
    report_dir: Path
    pure_class_examples: bool = True
    plot: PlotConfig = PlotConfig()

    @property
    def report_path(self) -> Path:
        return resolve_path(self.report_dir)


class ModelConfig(StrictModel):
    # Must match src.models.factory.SUPPORTED_MODELS (checked by tests/test_models.py); listed here
    # rather than imported so that loading configs never requires torch
    name: Literal[
        "resnet18",
        "resnet34",
        "resnet50",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "efficientnet_b0",
        "shufflenet_v2_x0_5",
        "shufflenet_v2_x1_0",
        "slim_resnet",
    ]
    pretrained: bool = False
    # Remove the early downsampling designed for 224x224 photos, so small inputs keep their detail
    # (see src.models.factory.build_model)
    small_input_stem: bool = True
    # slim_resnet only: channels of the first stage (doubled at each stage) and blocks per stage
    width: int = Field(default=64, ge=4)
    depths: list[int] = Field(default=[2, 2, 2, 2], min_length=4, max_length=4)

    @model_validator(mode="after")
    def check_slim_resnet_options(self) -> Self:
        if self.name != "slim_resnet" and (self.width != 64 or self.depths != [2, 2, 2, 2]):
            raise ValueError("width and depths only apply to slim_resnet")
        if min(self.depths) < 1:
            raise ValueError("every stage needs at least one block")
        if self.name == "slim_resnet" and self.pretrained:
            raise ValueError("slim_resnet has no pretrained weights")
        return self


class PosWeightConfig(StrictModel):
    # "none": plain BCE; "raw": negatives; "sqrt": square root of that ratio (damped)
    mode: Literal["none", "raw", "sqrt"] = "none"
    # Upper bound applied after damping; None means no cap
    max_value: float | None = Field(default=None, gt=0)


class TrainConfig(StrictModel):
    # MLflow experiment that groups every run of this project
    experiment_name: str
    # References resolved by name: conf/split/<split>.yaml and conf/model/<model>.yaml
    split: str
    model: str

    encoding: Literal["onehot", "scaled"] = "onehot"
    augment: bool = False
    pos_weight: PosWeightConfig = PosWeightConfig()

    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    scheduler: Literal["none", "cosine"] = "cosine"
    # Epochs of linear learning-rate warmup before the schedule, against the instability of early epochs
    warmup_epochs: int = Field(default=0, ge=0)
    # Stop when the monitored metric has not improved for this many epochs; None disables it
    early_stopping_patience: int | None = Field(default=None, gt=0)
    # Threshold-free metric used to pick the best checkpoint (decision thresholds are tuned later)
    monitor: Literal["val_macro_ap", "val_macro_f1"] = "val_macro_ap"
    # Which weights to keep: "best" = the epoch with the highest monitor value, "last" = the final epoch.
    checkpoint: Literal["best", "last"] = "best"

    seed: int = 42
    num_workers: int = Field(default=4, ge=0)

    @model_validator(mode="after")
    def check_warmup(self) -> Self:
        if self.warmup_epochs >= self.epochs:
            raise ValueError("warmup_epochs must be smaller than epochs")
        return self

    @model_validator(mode="after")
    def check_checkpoint_protocol(self) -> Self:
        if self.checkpoint == "last" and self.early_stopping_patience is not None:
            raise ValueError(
                "checkpoint 'last' cannot be combined with early stopping: training would stop before the "
                "learning rate schedule finishes, so the kept model would never go through the annealing phase"
            )
        return self


class FloatRange(StrictModel):
    low: float = Field(gt=0)
    high: float = Field(gt=0)
    log: bool = True

    @model_validator(mode="after")
    def check_order(self) -> Self:
        if self.low >= self.high:
            raise ValueError("low must be smaller than high")
        return self


class SearchSpace(StrictModel):
    # Architecture (slim_resnet): drives FLOPs
    width: list[int] = Field(min_length=1)
    depth: list[int] = Field(min_length=1)  # blocks in every stage
    small_input_stem: list[bool] = Field(min_length=1)
    # Training: drives accuracy
    learning_rate: FloatRange
    weight_decay: FloatRange
    warmup_epochs: list[int] = Field(min_length=1)


class SearchConfig(StrictModel):
    study_name: str
    # MLflow experiment for the trials, kept apart from the main experiment
    experiment_name: str
    # Every setting not searched (split, encoding, augmentation, batch size, ...) comes from this
    # experiment config: conf/experiments/<base_experiment>.yaml
    base_experiment: str
    # Shorter than the full schedule: the search screens candidates, finalists are retrained in full
    epochs: int = Field(gt=0)
    n_trials: int = Field(gt=0)
    seed: int = 0
    # Each trial is scored by its validation errors averaged over its last `last_k` epochs
    last_k: int = Field(default=3, gt=0)
    space: SearchSpace


def load_config[ConfigT: BaseModel](path: str | Path, model: type[ConfigT]) -> ConfigT:
    """Load a YAML file and validate it against a schema."""
    full_path = resolve_path(path)
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found: {full_path}")

    with open(full_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return model.model_validate(data)


def named_config_path(group: str, name: str) -> Path:
    """Path of a named config, e.g. ("experiments", "wafer_augment") -> <root>/conf/experiments/wafer_augment.yaml."""
    return resolve_path(CONF_DIR / group / f"{name}.yaml")


def load_named_config[ConfigT: BaseModel](group: str, name: str, model: type[ConfigT]) -> ConfigT:
    """Configs refer to each other by name; this resolves a name to conf/<group>/<name>.yaml."""
    return load_config(named_config_path(group, name), model)


def load_dataset_config(name: str) -> DatasetConfig:
    return load_named_config("dataset", name, DatasetConfig)


def load_split_config(name: str) -> SplitConfig:
    return load_named_config("split", name, SplitConfig)


def load_model_config(name: str) -> ModelConfig:
    return load_named_config("model", name, ModelConfig)


def load_experiment_config(name: str) -> TrainConfig:
    return load_named_config("experiments", name, TrainConfig)
