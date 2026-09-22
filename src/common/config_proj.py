import logging
import os
import re
from pathlib import Path
from typing import Any, get_origin

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.common.path import get_project_root

# --- Get the Absolute Path of the project and ".env" file ---
PROJECT_ROOT = get_project_root()
ENV_PATH = PROJECT_ROOT / ".env"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class AppSettings(BaseSettings):
    # --- Environment Settings ---
    env_mode: str = "dev"

    # --- Get Single Token ---
    db_mysql_token: str | None = Field(None, validation_alias="DB_MYSQL_TOKEN")
    huggingface_token: str | None = Field(None, validation_alias="HUGGINGFACE_TOKEN")

    # --- Get Group Token ---
    gemini_tokens: list[str] = Field(default_factory=list)

    # --- Get Config ---
    directories: dict[str, Path] = {}
    services: dict[str, Any] = {}

    model_config = SettingsConfigDict(env_file=ENV_PATH, env_file_encoding="utf-8", extra="ignore", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def scan_multi_tokens(cls, data: Any) -> Any:
        """
        Multiple-token scaner.
        """
        if not isinstance(data, dict):
            data = {}

        for field_name, field_info in cls.model_fields.items():
            annotation_ori = get_origin(field_info.annotation)
            is_list = annotation_ori is list or field_info.annotation is list

            if field_name.endswith("_tokens") and is_list:
                # get the strings equal "<prefix><number>"
                prefix = field_name.upper().replace("TOKENS", "TOKEN_")
                pattern = re.compile(rf"^{prefix}(\d+)$")

                found_tokens = []
                temp = {}
                for env_key, env_value in os.environ.items():
                    match = pattern.match(env_key)
                    if match:
                        index = int(match.group(1))
                        temp[index] = env_value

                sorted_indices = sorted(temp.keys())
                for idx in sorted_indices:
                    found_tokens.append(temp[idx])

                if found_tokens:
                    data[field_name] = found_tokens
        return data

    @field_validator("directories")
    @classmethod
    def resolve_directories(cls, v: dict[str, Path]) -> dict[str, Path]:
        resolved_paths = {}
        for key, path in v.items():
            # remove '../' and '//' using resolve function
            absolute_path = (PROJECT_ROOT / path).resolve()
            resolved_paths[key] = absolute_path

            if not absolute_path.exists():
                msg = f"Directory not found: {absolute_path}"
                logger.error(msg)
        return resolved_paths


def init_settings(config_path: str = "configs/config.yaml") -> AppSettings:
    yaml_full_path = PROJECT_ROOT / config_path
    if not yaml_full_path:
        msg = f"Config file not found at: {yaml_full_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    with open(yaml_full_path, encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f) or {}

    return AppSettings(
        directories={k: Path(v) for k, v in yaml_data.get("directories", {}).items()},
        services=yaml_data.get("services", {}),
    )  # type: ignore
