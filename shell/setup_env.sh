#!/bin/bash

ENV_YML="./bak/environment.yml"

# Obtain the environment name from ".yml" file
ENV_NAME=$(grep "^name:" "$ENV_YML" | awk '{print $2}')

# Check if conda exists
if ! command -v conda &> /dev/null; then
    echo ">>> Conda is not installed, please install Anaconda or Miniconda first!"
    exit 1
fi

# Optional --force flag to rebuild environment
if [[ "$1" == "--force" ]]; then
    echo ">>> Removing existing environment '$ENV_NAME' ..."
    conda env remove -n "$ENV_NAME" -y
fi

# Create env if not exists
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">>> Conda environment '$ENV_NAME' already exists, skip creating it."
else
    echo ">>> Create Conda environment '$ENV_NAME' ..."
    conda env create -f "$ENV_YML"
fi

# Initialize pre-commit
echo ""
echo ">>> Install and initialize pre-commit ..."
make init-hooks

echo ""
echo ">>> Done! To activate, run:"
echo ">>>   conda activate $ENV_NAME"
