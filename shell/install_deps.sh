#!/usr/bin/env bash
#
# Install the project's Python dependencies into the active environment, the same way everywhere:
# "make init" and "make conda-dev" run this script locally, and CI runs it with TORCH=cpu, so every
# push also tests the setup a user gets from a fresh clone.
#
#   1. pip itself, from its hash-pinned URL (requirements-pip.txt)
#   2. the development lock file (requirements-dev.txt: a superset that already includes requirements.txt)
#   3. the project in editable mode, so that 'src' is importable from any directory (scripts, tests, notebooks)
#
# TORCH selects the PyTorch build:
#   TORCH=cuda (default)  the lock file as it is: the CUDA build from PyPI (Linux x86_64 / WSL2; it also runs on a CPU)
#   TORCH=cpu             the CPU builds of the same torch and torchvision versions: a few hundred MB instead
#                         of about 4 GB of CUDA libraries, for machines without an NVIDIA GPU
#
# Usage:
#   bash shell/install_deps.sh
#   TORCH=cpu bash shell/install_deps.sh
#
# Use "set -euo pipefail":
#   -e: if any command fails, terminate immediately
#   -u: throw an error immediately when an undefined variable is used
#   -o pipefail: a failure at any stage of the pipeline is considered a failure of the entire instruction

set -euo pipefail

TORCH="${TORCH:-cuda}"
LOCK="requirements-dev.txt"

if [[ "$TORCH" != "cuda" && "$TORCH" != "cpu" ]]; then
    echo "!! Unknown TORCH='$TORCH': use 'cuda' (default) or 'cpu'" >&2
    exit 1
fi

echo ">>> [1/3] Install pip from its hash-pinned URL (requirements-pip.txt)"
python -m pip install -r requirements-pip.txt

case "$TORCH" in
    cuda)
        echo ">>> [2/3] Install dependencies ($LOCK: a superset that already includes requirements.txt)"
        python -m pip install -r "$LOCK"
        ;;
    cpu)
        echo ">>> [2/3] Install dependencies ($LOCK) with the CPU builds of torch and torchvision"
        mapfile -t torch_pins < <(grep -E '^(torch|torchvision)==' "$LOCK")
        if [[ ${#torch_pins[@]} -ne 2 ]]; then
            echo "!! Expected the torch and torchvision pins in $LOCK, found: ${torch_pins[*]:-none}" >&2
            exit 1
        fi
        # The pin "torch==X" accepts the CPU build "X+cpu", so the rest of the lock file installs on top
        # of it unchanged, once its CUDA packages are left out.
        python -m pip install --index-url https://download.pytorch.org/whl/cpu "${torch_pins[@]}"
        filtered="$(mktemp)"
        trap 'rm -f "$filtered"' EXIT
        grep -v -E '^(nvidia-|cuda-|triton==)' "$LOCK" > "$filtered"
        python -m pip install -r "$filtered"
        ;;
esac

echo ">>> [3/3] Install the project in editable mode (makes 'src' importable from any directory)"
python -m pip install --no-deps -e .
