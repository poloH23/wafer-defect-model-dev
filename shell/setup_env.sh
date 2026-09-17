#!/bin/bash
set -euo pipefail

ENV_YML="./bak/environment.yml"

# Get the environment name from .yml file
ENV_NAME=$(grep "^name:" "$ENV_YML" | awk '{print $2}')

# Check if conda exists
if ! command -v conda &> /dev/null; then
    echo ">>> Conda is not installed, please install Anaconda or Miniconda first!"
    exit 1
fi

# '--force' argument: rebuild environment
if [[ "${1:-}" == "--force" ]]; then
    echo ">>> Removing existing environment '$ENV_NAME' ..."
    conda env remove -n "$ENV_NAME" -y
fi

# Create the environment only if it does not exist.
# Note: The environment.yml file now only pins the Python version and pip itself
# The actual list of components is managed by requirements.txt / requirements-dev.txt
# Avoid having conda and pip manage separate copies of the same package
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">>> Conda environment '$ENV_NAME' already exists, skip creating it."
else
    echo ">>> Create Conda environment '$ENV_NAME' ..."
    conda env create -f "$ENV_YML"
fi

# Perform dependency installation, pre-commit hook installation, and baseline checks the newly created environment
# Use 'conda run -n' to explicitly specify the execution environment
echo ""
echo ">>> Installing dependencies and pre-commit hooks inside '$ENV_NAME' ..."
conda run -n "$ENV_NAME" bash shell/init_project.sh

echo ""
echo ">>> Done! To activate, run:"
echo ">>> conda activate $ENV_NAME"
