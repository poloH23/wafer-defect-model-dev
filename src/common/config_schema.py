"""
Schemas and loaders for task configs stored under conf/.
The schemas are generic (they discribe what any image classification dataset, split or EDA run needs);
the project-specific values lives only in the YAML files.

Layout:
    conf/dataset/<name>.yaml  : facts about the a dataset, shared by every tasks
    conf/split/<name>.yaml    : parameters of one split version
    conf/eda/<name>.yaml      : parameters of one EDA run
"""

from pathlib import Path
from typing import Self

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
    class_name: list[str] = Field(min_length=1)  # notes: order matters

    @property
    def source_path(self) -> Path:
        return resolve_path(self.source)

    @property
    def class_name_map(self) -> dict[str, str]:
        """Map the generic ids produced by analysis module (class_0, class_1, ...) to readable names."""
        return {f"class_{i}": name for i, name in enumerate(self.class_name)}


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


def load_config[ConfigT: BaseModel](path: str | Path, model: type[ConfigT]) -> ConfigT:
    """Load a YAML file and validate it against a schema."""
    full_path = resolve_path(path)
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found: {full_path}")

    with open(full_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return model.model_validate(data)


def load_dataset_config(name: str) -> DatasetConfig:
    """Task configs refer to their dataset by name; this resolves it to conf/dataset/<name>.yaml."""
    return load_config(CONF_DIR / "dataset" / f"{name}.yaml", DatasetConfig)
