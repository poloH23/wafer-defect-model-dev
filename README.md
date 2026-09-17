# Wafer Defect Classification Model Development Project

<!-- TODO: Briefly describe this project -->

[![CI](https://github.com/poloH23/wefer-defect-model-dev/actions/workflows/ci.yml/badge.svg)](https://github.com/poloH23/wefer-defect-model-dev/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


## Conda Environment Construction

### Prerequisites

- Miniconda or Anaconda
- Python 3.12 (Automatically set up by running `make conda-dev`; no separate manual installation is required.)

### Quick Start

```bash
git clone https://github.com/poloH23/wefer-defect-model-dev.git
cd wefer-defect-model-dev
make conda-dev
conda activate torch
```

`make conda-dev` executes the following steps in order, so there is no need to run them manually one by one:

1. Create a clean conda environment based on `bak/environment.yml` (locking only the Python version).
2. Install all runtime dependencies (`requirements.txt`) and development dependencies (`requirements-dev.txt`) in the new environment.
3. Install pre-commit Git hook
4. Perform a baseline check on the entire project.


### Verification

```bash
make test
```

Try running `git commit` and verify that the pre-commit hook automatically triggers formatting and linting checks.


## Commands

Run `make help` to view the full list (grouped by category and with descriptions).

|Command|Instruction|
|---|---|
|`make conda-dev`|Create conda environmnet and perform the initialization (**Recommended for initial setup**)|
|`make conda-custom ENV=myenv`|Create a standalone conda environment with a specified name (functionally identical, but allows for a custom environment name)|
|`make init`|Install dependencies and set up hooks in the **currently active** environment (without creating a new conda environment)|
|`make test`|Execute the test suite and display coverage|
|`make code-format`|Use Ruff to format code and automatically fix linting issues|
|`make lint`|Run all pre-commit checks|
|`make requirments`|Re-lock the versions in `requirements.txt` / `requirements-dev.txt` based on the `.in` files|
|`make requirments-bootstrap`|Scan the source code for import statements and regenerate `requirements.in`|
|`make requirments-refresh`|Combine the two steps above into a single operation (rescan + re-lock version)|
|`make clean`|Delete cache files (such as `__pycache__`, `.pytest_cache`, etc.—files that are safe and reproducible)|


## Dependency Package Management

This project uses `pip-tools` to split dependency declarations into two levels:
- **`requirements.in` / `requirements-dev.in`**: Manually maintained lists of "direct" dependencies, listing only the packages actually used by the code.
- **`requirements.txt` / `requirements-dev.txt`**: Generated via `make requirements` after automatically resolving and locking versions; **do not edit manually**, and ensure both files are committed to version control.

When adding a new dependency:

1. Modify the corresponding `.in` file (or run `make requirements-bootstrap` to rescan the source code).
2. Run `make requirements` to re-lock the versions.
3. Commit both the modified `.in` and `.txt` files.

For some packages, the import name differs from the PyPI package name (e.g., `import cv2` corresponds to `opencv-python-headless`); since automated scanning tools may not always accurately identify these cases, it is recommended to verify the contents of `requirements.in` after adding them.


## Continuous Integration

Every push or pull request triggers GitHub Actions (defined in `.github/workflows/ci.yml`), which includes:

- **pre-commit checks**: Uses the same configuration as the local `.pre-commit-config.yaml`, thereby avoiding discrepancies where local checks pass but CI fails.
- **Test suite**: Executes pytest and generates a coverage report.


## Project Structure

<!-- TODO:  -->

## License

This project is licensed under the [Apache License 2.0](LICENSE)
