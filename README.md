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


## Data Qality Findings

Issues found in the official dataset, and how they are handeled. The raw file is never modified:
every rule below is applied in code and configured under 'conf/'.

- **Extra duplicate images.** 1,024 images are byte-for-byte copies of another image with the same labels.
  1,000 of them form a complete second copy of the Center+Edge_Loc+Scratch combination, which explains
  why the documentation lists 2,000 samples for that combination and 1,000 for every other one. The
  remaining 24 belong to Random (16) and Near_Full (8). A first random split left 157 validation and
  174 test images with an identical copy in the training split, inflating the validation scores.
  Split 'v2' keeps a single copy of each image, and every reported result uses it.
- **Undocumented pixel values.** The documentation lists three pixel values (0 = empty, 1 = pass,
  2 = fail), but the file contains 214 pixels with the value 3 in 105 images, all labeled Random (97)
  or Near_Full (8): the two classes whose labels the authors corrected after release.


## Limitations

### Data

- **Partly synthetic data.** The dataset authors generated part of the samples with a generative
  adversarial network to balance the classes, and synthetic samples are not marked.
- **Artificially balanced class distribution.** Most label combinations have exactly 1,000 samples.
  Real production lines are dominated by normal wafers, so precision in production would be lower
  than reported here, and decision thresholds would need to be recalibrated on real data.
- **Near_Full and Random never co-occur with other defects.** The model has never seen them in
  mixed patterns, so its behavior on such combinations is untested.
- **Edge_Loc / Edge_Ring label ambiguity.** By definition, Edge_Ring is a defect along the whole wafer
  edge and Edge_Loc a defect on part of it. A rule-based edge coverage measure (the share of the
  circumference whose outermost band consists mostly of failed dies) separates the two labels well
  overall (ROC AUC 0.987–0.996 for sector thresholds from 0.4 to 0.6), yet about 3% of Edge_Loc samples
  cover nearly the entire edge, some of them completely. All 14 validation samples where the baseline
  predicted Edge_Ring instead of Edge_Loc come from this group, in the top 4% of Edge_Loc coverage under
  every threshold tested. In this region the labels contradict the class definitions, so part of the
  remaining error reflects how the data was labeled rather than the model's capacity.

### Evaluation
- **Small rare-class evaluation sets.** After deduplication, Near_Full has 141 samples: 14 in
  validation and 15 in test. A single test error changes Near_Full recall by about 6.7 percentage points.
- **A single random split.** All splits come from the same source and the same distribution.
  A split by time, production line or fab would give a more realistic estimate, but this
  information is not available in the dataset.

### Scope

- **Fixed input format.** The model is trained on 52×52 maps from a single source; other die
  layouts or resolutions would require retraining.
- **No deployment evaluation.** Inference latency and robustness to distribution shift are out
  of scope here and are covered by the follow-up deployment project.


## Project Structure

<!-- TODO:  -->

## License

This project is licensed under the [Apache License 2.0](LICENSE)
