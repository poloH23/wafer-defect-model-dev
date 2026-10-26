#!/usr/bin/env bash
#
# Project initialization, in the active environment:
# Install dependencies and the project itself (shell/install_deps.sh), set up per-commit hooks,
# and perform a project-wide baseline check
# Usage:
#   make init               # in the environment that is active now (TORCH=cpu for the CPU-only PyTorch)
#   make conda-dev          # creates the conda environment first, then runs this script inside it
#
# Use "set -euo pipefail":
#   -e: if any command fails, terminate immediately
#   -u: throw an error immediately when an undefined variable is used
#   -o pipefail: a failure at any stage of the pipeline is considered a failure of the entire instruction

set -euo pipefail

echo ">>> [1/3] Install dependencies and the project in editable mode (TORCH=${TORCH:-cuda})"
bash "$(dirname "${BASH_SOURCE[0]}")/install_deps.sh"

echo ">>> [2/3] Mount pre-commit git hook"
pre-commit install

echo ">>> [3/3] Perform a baseline check for the entire project"
if ! pre-commit run --all-files --show-diff-on-failure; then
    echo ""
    echo "!! pre-commit: It is normal to encounter issues within the existing code during the initial integration phase"
    echo "!! Please review the automatic corrections in the output above; once confirmed, execute 'git add' and 'git commit'"
fi

echo ""
echo ">>> Project initialization complete"
