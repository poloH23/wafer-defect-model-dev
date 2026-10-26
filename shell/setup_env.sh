#!/usr/bin/env bash
#
# Create (or reuse) the project's conda environment, then initialize the project inside it
# (shell/init_project.sh). Conda provides only Python and pip: every package comes from the pip lock
# files (requirements.txt / requirements-dev.txt), so conda and pip never manage two copies of the
# same package.
#
# Usage (through make; the variables are passed on as environment variables):
#   make conda-dev                        # environment "wafer-defect-model-dev", CUDA build of PyTorch
#   make conda-dev ENV=my-env TORCH=cpu   # another name; CPU-only PyTorch, a much smaller download
#   make conda-dev FORCE=1                # remove the environment and rebuild it
#
# Python comes from conda-forge only (--override-channels), whatever channels the user has configured;
# Anaconda's "defaults" channel has terms of service of its own.

set -euo pipefail

ENV_NAME="${ENV_NAME:-wafer-defect-model-dev}"
PYTHON_VERSION="3.12"  # keep in line with requires-python in pyproject.toml
TORCH="${TORCH:-cuda}"
FORCE="${FORCE:-}"

if ! command -v conda >/dev/null 2>&1; then
    echo "!! Conda is not installed: install Miniforge (or Miniconda / Anaconda) first" >&2
    exit 1
fi

# awk reads the whole list and decides at the end, so no early exit can upset "pipefail"
env_exists() {
    conda env list | awk -v name="$ENV_NAME" '$1 == name { found = 1 } END { exit !found }'
}

if [[ -n "$FORCE" ]] && env_exists; then
    echo ">>> Removing the existing environment '$ENV_NAME' (FORCE=1) ..."
    conda remove -y -n "$ENV_NAME" --all
fi

if env_exists; then
    echo ">>> Conda environment '$ENV_NAME' already exists: reusing it (FORCE=1 rebuilds it)"
else
    echo ">>> Create conda environment '$ENV_NAME' (Python $PYTHON_VERSION from conda-forge) ..."
    conda create -y -n "$ENV_NAME" --override-channels -c conda-forge "python=$PYTHON_VERSION" pip
fi

# Install dependencies and pre-commit hooks, and run the baseline check, inside the environment.
# "conda run -n" names the environment explicitly; --no-capture-output streams pip's progress
# instead of printing everything at the end, which matters for a download of several GB.
echo ""
echo ">>> Installing dependencies and pre-commit hooks inside '$ENV_NAME' (TORCH=$TORCH) ..."
conda run --no-capture-output -n "$ENV_NAME" env TORCH="$TORCH" bash shell/init_project.sh

echo ""
echo ">>> Done! To activate, run:"
echo ">>> conda activate $ENV_NAME"
