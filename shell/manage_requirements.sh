#!/usr/bin/env bash
#
# Requirement Management:
# Re-lock 'requirements.txt' and 'requirements-dev.txt' from the '.in' files, and compare the code's
# imports with 'requirements.in' without changing it.
#
# Usage:
#   bash shell/manage_requirements.sh compile  # re-lock versions and generate .txt files based on .in files
#   bash shell/manage_requirements.sh check    # compare the imports of src/ and scripts/ with requirements.in (read-only)
#
# compile passes any further options to the requirements-dev.txt compile, e.g.
#   bash shell/manage_requirements.sh compile --upgrade-package pillow
# releases pillow's pin there; requirements.txt then follows through -c requirements-dev.txt.
#
# requirements.in is curated by hand (no versions, except deliberate pins such as torch, each with its
# reason), so nothing here writes to it: an import scanner pins every package, maps some imports to the
# wrong distribution (cv2) and picks up test-only imports (pytest).

set -euo pipefail

# Ensure the pip-compile and pipreqs tools are installed; if not, install them based on requirements-dev.in
# Direct installation (the .in file is itself a valid pip requirements file format; it works fine without compilation)
ensure_dev_tools() {
    if ! command -v pip-compile >/dev/null 2>&1 || ! command -v pipreqs >/dev/null 2>&1; then
       echo ">>> pip-compile / pipreqs is not yet installed, bootstrapping from 'requirements-dev.in' first"
       pip install -r requirements-dev.in
    fi
}

# One distribution name per line, lower-case with "-", so that "PyYAML" / "scikit_learn" match "pyyaml" / "scikit-learn"
normalize() {
    sed -E 's/[[:space:]]+//g' | tr 'A-Z_' 'a-z-' | sed '/^$/d' | sort -u
}

check() {
    ensure_dev_tools
    echo ">>> Compare the third-party imports of src/ and scripts/ with requirements.in (no file is changed)"
    local scanned declared
    scanned="$(pipreqs . --print --mode no-pin --ignore tests,notebooks,.venv | normalize)"
    declared="$(sed -E 's/#.*//; s/[[:space:]]*[<>=!~;[].*//' requirements.in | normalize)"
    echo ""
    echo ">>> Imported, but not declared in requirements.in (add the right distribution if it is a real dependency):"
    comm -23 <(printf '%s\n' "$scanned") <(printf '%s\n' "$declared") | sed '/^$/d; s/^/    /'
    echo ">>> Declared, but not imported directly (keep it if something needs it indirectly):"
    comm -13 <(printf '%s\n' "$scanned") <(printf '%s\n' "$declared") | sed '/^$/d; s/^/    /'
    echo ""
    echo ">>> Import names are mapped to distributions by guesswork (e.g. cv2 is opencv-python-headless here): review the lists, do not copy them"
}

compile_lock() {
    ensure_dev_tools
    echo ">>> Recompile requirements-dev.txt (requirements.in + requirements-dev.in, resolved together)"
    pip-compile requirements.in requirements-dev.in -o requirements-dev.txt --constraint constraints.txt "$@"
    echo ">>> Recompile requirements.txt (constrained by requirements-dev.txt, so both pin the same versions)"
    pip-compile -c requirements-dev.txt requirements.in -o requirements.txt --constraint constraints.txt
    echo ">>> requirements-dev.txt / requirements.txt have been updated"
}

case "${1:-}" in
    compile) shift; compile_lock "$@" ;;
    check) check ;;
    *)
        echo "Usage: $0 {compile|check}"
        exit 1
        ;;
esac
