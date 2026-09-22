import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def get_project_root(anchor_file: str = ".env") -> Path:
    """
    Automatically search upwards from the current file for directories with "anchor_file" as the root directory.
    """
    load_dotenv()
    current_path = Path(__file__).resolve()

    # search upwards until '.env' file is found or till the root directory of the disk
    for parent in current_path.parents:
        if (parent / anchor_file).exists():
            os.environ["PROJECT_ROOT"] = str(parent)
            load_dotenv(dotenv_path=parent / anchor_file)
            return parent

    # if can not find the "anchor_file", return the path upwards the parent of this file (assume the structure is fixed)
    fallback_root = current_path.parent.parent.parent
    os.environ["PROJECT_ROOT"] = str(fallback_root)
    return fallback_root


def setup_python_path() -> None:
    """
    Integrate the design of get_module and ensure the modules under the "src" can be imported.
    """
    root = get_project_root()
    root_path = str(root)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)


def get_path_by_env(key: str) -> Path:
    """
    Get the path from environment variables and ensure the ensure cross-platform compatibility.
    """
    root = get_project_root()
    path_suffix = os.getenv(key)

    if not path_suffix:
        raise ValueError(f"Environment variable '{key}' is not set!")

    full_path = (root / path_suffix.lstrip("/\\")).resolve()
    return full_path


def validate_paths(*env_vars: str) -> None:
    """
    Check whether the path exists or not.
    """
    for var in env_vars:
        path = get_path_by_env(var)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        print(f"Verified path: {path}")
