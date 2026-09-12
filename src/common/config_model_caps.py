import logging
from pathlib import Path

import yaml
from pydantic import BaseModel

# --- Get the Absolute Path of the project ---
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TaskConfig(BaseModel):
    metrics: list[str]


class ModelConfig(BaseModel):
    tasks: list[str]


class CapabilitiesSchema(BaseModel):
    tasks: dict[str, TaskConfig]
    models: dict[str, ModelConfig]


class ModelCapsManager:
    """
    Specifically manage model capabilities and estimation indicators. (Singleton Pattern)
    """

    def __init__(self, config_path: str = "configs/model_capabilities.yaml"):
        self.config_path = PROJECT_ROOT / Path(config_path)
        self._data: CapabilitiesSchema | None = None
        self._load_config()

    def _load_config(self):
        try:
            if not self.config_path.exists():
                msg = f"Config file not found at: {self.config_path}"
                logger.error(msg)
                raise FileNotFoundError(msg)

            with open(self.config_path, encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)
                self._data = CapabilitiesSchema(**yaml_data)
        except Exception as e:
            self._data = None
            msg = f"Exception while loading model capability config: {e}"
            logger.error(msg)
            return None

    def get_supported_metrics(self, model_name: str) -> set[str | None]:
        """
        Get supported metrics of the specific model.
        """
        if not self._data:
            return set()

        if model_name not in self._data.models:
            msg = f"Unsuppored model: {model_name}"
            logger.error(msg)
            return set()

        tasks = self._data.models[model_name].tasks

        supported_metrics = set()
        for task in tasks:
            if task in self._data.tasks:
                supported_metrics.update(self._data.tasks[task].metrics)
        return supported_metrics

    def validate(self, model_name: str, metrics: str) -> bool:
        """
        Validation core.
        """
        if not self._data:
            return False

        supported = self.get_supported_metrics(model_name)
        if not supported:
            return False
        return metrics in supported

    def print_summary(self) -> None:
        """
        Show the model capability registry.
        """
        if not self._data:
            print("No mode capability config loaded.")
            return

        print("\n" + "=" * 100)
        print("Model Capability Registry")
        print("=" * 100)

        # --- Section 1: Task Definition ---
        print("\n[  Section 1: Task Definition  ]")
        print(f"{'Task Name':<25} | {'Supported Metrics'}")
        print("-" * 100)

        for task_name, task_config in sorted(self._data.tasks.items()):
            metrics_str = ", ".join(sorted(task_config.metrics))
            print(f"{task_name:<25} | {metrics_str}")

        # --- Section 2: Model Capability ---
        print("\n[  Section 2: Model Capability  ]")
        print(f"{'Model Name':<15} | {'Tasks':<30} | {'All Avalibel Metrics'}")
        print("-" * 100)

        for name, config in sorted(self._data.models.items()):
            tasks = ", ".join(config.tasks)
            all_metrics = self.get_supported_metrics(name)
            metrics = ", ".join(sorted(list(all_metrics)))  # type:ignore
            print(f"{name:<15} | {tasks:<30} | {metrics}")
        print("\n" + "=" * 100 + "\n")
        return


model_capabilities = ModelCapsManager()
