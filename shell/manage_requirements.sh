#!/usr/bin/env bash
#
# Requirement Management:
# Centrally manage the re-scanning and re-locking of 'requirements.in' and 'requirements.txt' files.
#
# Usage:
#   bash shell/manage_requirements.sh bootstrap  # scan the source code for import statements and regenerate 'requirements.in'
#   bash shell/manage_requirements.sh compile    # re-lock versions and generate .txt files based on .in files
#   bash shell/manage_requirements.sh refresh    # Perform the two steps above in order

set -euo pipefail

# Ensure the pip-compile and pipreqs tools are installed; if not, install them based on requirements-dev.in
# Direct installation (the .in file is itself a valid pip requirements file format; it works fine without compilation)
ensure_dev_tools() {
    if ! command -v pip-compile >/dev/null 2>&1 || ! command -v pipreqs >/dev/null 2>&1; then
       echo ">>> pip-compile / pipreqs is not yet installed, bootstrapping from 'requirements-dev.in' first"
       pip install -r requirements-dev.in
    fi
}

bootstrap() {
    ensure_dev_tools
    echo ">>> Rescan the source code for imports and update requirements.in"
    pipreqs . --savepath requirements.in --force
    echo ">>> 'requirements.in' has been updated; please check the diff before committing"
}

compile_lock() {
    ensure_dev_tools
    echo ">>> Recompile requirements-dev.txt (requirements.in + requirements-dev.in, resolved together)"
    pip-compile requirements.in requirements-dev.in -o requirements-dev.txt --constraint constraints.txt
    echo ">>> Recompile requirements.txt (constrained by requirements-dev.txt, so both pin the same versions)"
    pip-compile -c requirements-dev.txt requirements.in -o requirements.txt --constraint constraints.txt
    echo ">>> requirements-dev.txt / requirements.txt have been updated"
}

case "${1:-}" in
    bootstrap) bootstrap ;;
    compile) compile_lock ;;
    refresh) bootstrap; compile_lock ;;
    *)
        echo "Usage: $0 {bootstrap|compile|refresh}"
        exit 1
        ;;
esac
