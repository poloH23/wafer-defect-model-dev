#!/bin/bash

# Default target is current project root
TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
SHOW_ONLY=0
VERBOSE=0
EXCLUDE_DIRS=()

show_help() {
  echo "Usage: gitkeep.sh [options]"
  echo "  -c              Remove all .gitkeep files"
  echo "  -s              Show preview only"
  echo "  -v              Verbose mode"
  echo "  -e <dir1,dir2>  Exclude specific directories"
  echo "  -h              Show help"
}

while getopts "csve:h" opt; do
  case $opt in
    c)
      echo "[Clean] Removing all .gitkeep files..."
      find "$TARGET_DIR" -type f -name ".gitkeep" -exec rm {} +
      echo ".gitkeep files removed."
      exit 0
      ;;
    s) SHOW_ONLY=1 ;;
    v) VERBOSE=1 ;;
    e) IFS=',' read -r -a EXCLUDE_DIRS <<< "$OPTARG" ;;
    h) show_help; exit 0 ;;
    *) show_help; exit 1 ;;
  esac
done

echo "Searching for empty directories in: $TARGET_DIR"

EXCLUDE_FIND=""
for dir in "${EXCLUDE_DIRS[@]}"; do
  EXCLUDE_FIND="$EXCLUDE_FIND -path \"$TARGET_DIR/$dir\" -prune -o"
done

eval find "$TARGET_DIR" -type d -empty $EXCLUDE_FIND -print | while read -r folder; do
  if [ "$SHOW_ONLY" -eq 1 ]; then
    echo "[Preview] $folder"
  else
    touch "$folder/.gitkeep"
    [ "$VERBOSE" -eq 1 ] && echo "Added .gitkeep to: $folder"
  fi
done

[ "$SHOW_ONLY" -eq 1 ] && echo "Preview done." || echo ".gitkeep process completed!"

## Example Usages:
## ./gitkeep.sh                       → Add .gitkeep to all empty folders
## ./gitkeep.sh -v                    → Show and add .gitkeep to all empty folders
## ./gitkeep.sh -c                    → Clean all .gitkeep files
## ./gitkeep.sh -s -v                 → Show (verbose) only, no file created
## ./gitkeep.sh -d src -e logs,venv   → Specify target and exclude directories
