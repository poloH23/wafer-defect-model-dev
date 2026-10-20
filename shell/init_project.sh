#!/usr/bin/env bash
#
# Project initialization:
# Install dependencies, set up per-commit hooks, and perform a project-wide baseline check
# Usage:
#   make init
#
# Use "set -euo pipefail":
#   -e: if any command fails, terminate immediately
#   -u: throw an error immediately when an undefined variable is used
#   -o pipefail: a failure at any stage of the pipeline is considered a failure of the entire instruction

set -euo pipefail

echo ">>> [1/4] Install dependencies (requirements-dev.txt: a superset that already includes requirements.txt)"
pip install -r requirements-dev.txt

echo ">>> [2/4] Install the project in editable mode (makes 'src' importable anywhere)"
pip install -e .

echo ">>> [3/4] Mount pre-commit git hook"
pre-commit install

echo ">>> [4/4] Perform a baseline check for the entire project"
if ! pre-commit run --all-files --show-diff-on-failure; then
    echo ""
    echo "!! pre-commit: It is normal to encounter issues within the existing code during the initial integration phase"
    echo "!! Please review the automatic corrections in the output above; once confirmed, execute 'git add' and 'git commit'"
fi

echo ""
echo ">>> Project initialization complete"
