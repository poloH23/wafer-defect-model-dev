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
    """Paths in configs is relative to the project root, so thry work from any working directory."""
    path = Path(path)
    return path if path.is_absolute() else (get_project_root() / path).resolve()


class StrictModel(BaseModel):
    # extra="forbid": an unknown key (usually a type) is an error, not silently ignored
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

    @model_validator(mode="after")
    def check_class_names(self) -> Self:
        if len(self.class_names) != self.num_classes:
            raise ValueError(f"class_name has {len(self.class_names)} entries, expected num_classes = {self.num_classes}")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names contains duplicates.")
        return self

    @property
    def source_path(self) -> Path:
        return resolve_path(self.source)

    @property
    def class_name_map(self) -> dict[str, str]:
        """Map the generic ids produced by analysis module (class_0, class_1, ...) to readable names."""
        return {f"class_{i}": name for i, name in enumerate(self.class_names)}


class SplitConfig(StrictModel):
    dataset: str
    version: str
    seed: int
    val_ratio: float = Field(gt=0, lt=1)
    test_ratio: float = Field(gt=0, lt=1)
    min_stratum_count: int = Field(default=10, ge=1)
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
    name: Literal["resnet18", "resnet34", "resnet50"]
    pretrained: bool = False
    # Replace the ImageNet stem (7x7 conv stride 2 + max pool) with a 3x3 stride-1 conv, for small inputs
    small_input_stem: bool = True


class PosWeightConfig(StrictModel):
    # "none": plain BCE; "raw": negatives; "sqrt": square root of that ratio (damped)
    mode: Literal["none", "raw", "sqrt"] = "none"
    # Upper bound applied after damping; None means no cap
    max_value: float | None = Field(default=None, gt=0)


class TrainConfig(StrictModel):
    # MLflow experiment that groups every run of this project
    experiment_name: str
    # Reference resolved by name: conf/split/<split>.yaml and conf/model/<model>.yaml
    split: str
    model: str

    encoding: Literal["onehot", "scaled"] = "onehot"
    augment: bool = False
    pos_weight: PosWeightConfig = PosWeightConfig()

    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(gt=0)
    scheduler: Literal["none", "cosine"] = "cosine"
    # Stop when the monitored metric has not imporved for this many epochs; None diables it
    early_stopping_patience: int | None = Field(default=None, gt=0)
    # Threshold-free metric used to pick the best checkpoint (decision thresholds are tuned later)
    monitor: Literal["val_macro_ap", "val_macro_f1"] = "val_macro_ap"

    seed: int = 42
    num_workers: int = Field(default=4, ge=0)


def load_config[ConfigT: BaseModel](path: str | Path, model: type[ConfigT]) -> ConfigT:
    """Load a YAML file and validate it against a schema."""
    full_path = resolve_path(path)
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found: {full_path}")

    with open(full_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return model.model_validate(data)


def load_named_config[ConfigT: BaseModel](group: str, name: str, model: type[ConfigT]) -> ConfigT:
    """Configs refer to each other by name; this resolves a name to conf/<group>/<name>.yaml."""
    return load_config(CONF_DIR / group / f"{name}.yaml", model)


def laod_dataset_config(name: str) -> DatasetConfig:
    return load_named_config("dataset", name, DatasetConfig)


def load_split_config(name: str) -> SplitConfig:
    return load_named_config("split", name, SplitConfig)


def load_model_config(name: str) -> ModelConfig:
    return load_named_config("model", name, ModelConfig)
