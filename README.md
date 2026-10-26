# Wafer Defect Classification Model Development Project

> Language : *[繁體中文](README.zh-TW.md)*

Multi-label classification of semiconductor wafer maps (MixedWM38, 8 defect types), from data quality
checks to a deployable model: an evaluation protocol that measures its own noise, an error analysis
that traced the remaining errors to ambiguous labels, architecture comparison and multi-objective
search, efficiency benchmarks for three deployment targets, Grad-CAM explainability checked against a
control, and a verified, self-contained ONNX model. A follow-up project builds an inference service on
the delivered model.

- **98.76% of 3,700 held-out test maps** get all 8 labels right (46.0 ± 1.7 errors).
- **One model, ShuffleNetV2 1.0x, for three targets**: cloud GPU, CPU server (0.82 ms per map) and edge.
- **Selection rules committed before the test set was used**; negative results and the rules' flaws are
  reported, not hidden.
- **Delivered as one ONNX file** ([release v1.0.0](../../releases/tag/v1.0.0)) with a model card,
  verified against PyTorch on every validation map.

[![CI](https://github.com/poloH23/wafer-defect-model-dev/actions/workflows/ci.yml/badge.svg)](https://github.com/poloH23/wafer-defect-model-dev/actions/workflows/ci.yml)
[![Dependency audit](https://github.com/poloH23/wafer-defect-model-dev/actions/workflows/audit.yml/badge.svg)](https://github.com/poloH23/wafer-defect-model-dev/actions/workflows/audit.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


## Setup

### Prerequisites

- Linux x86_64, or WSL2 on Windows: the lock files pin Linux builds, including NVIDIA's CUDA libraries.
- conda: Miniforge is recommended; Miniconda and Anaconda work too, since the environment is created
  from conda-forge either way. Python 3.12 is installed into it automatically.
- GNU Make and Git.
- Optional: an NVIDIA GPU for training. Without one, PyTorch runs on the CPU: slower, but enough for the
  tests and the sanity check.

### Quick Start

```bash
git clone https://github.com/poloH23/wafer-defect-model-dev.git
cd wafer-defect-model-dev
make conda-dev          # without an NVIDIA GPU: make conda-dev TORCH=cpu
conda activate wafer-defect-model-dev
```

`make conda-dev` executes the following steps in order, so there is no need to run them manually one by one:

1. Create a conda environment named `wafer-defect-model-dev` (`ENV=...` to change it) with only Python 3.12
   and pip, from conda-forge. An existing environment of that name is reused; `FORCE=1` rebuilds it.
2. Install pip from its hash-pinned URL, then the development lock file (`requirements-dev.txt`, which
   includes `requirements.txt`). `TORCH=cpu` installs the CPU builds of the pinned PyTorch versions
   instead: a few hundred MB rather than about 4 GB of CUDA libraries.
3. Install the project itself in editable mode, so that `src` is importable from any directory
   (scripts, tests, notebooks).
4. Install the pre-commit Git hooks.
5. Perform a baseline check on the entire project.

Steps 2 and 3 are `shell/install_deps.sh`, which CI runs too (with `TORCH=cpu`), so every push also tests
this setup. In an environment that is already active, `make init` runs steps 2 to 5.

### Data

The dataset is not included in this repository. Download MixedWM38 from its authors' repository,
[Junliangwangdhu/WaferMap](https://github.com/Junliangwangdhu/WaferMap), unzip it, and place
`Wafer_Map_Datasets.npz` at `data/raw/Wafer_Map_Datasets.npz`. The fixed splits are committed under
`data/splits/`, and every run checks the data file's hash against them.

The dataset comes from J. Wang et al., "Deformable Convolutional Networks for Efficient Mixed-Type
Wafer Defect Pattern Recognition", *IEEE Transactions on Semiconductor Manufacturing*, 2020; it is
subject to its authors' terms.

### Verification

```bash
make test
make sanity-check   # once the dataset is in place: overfits one batch to check the training pipeline
```

Try running `git commit` and verify that the pre-commit hook automatically triggers formatting and linting checks.


## Commands

Run `make` (or `make help`) to list every command by workflow stage. Variables and their defaults are
at the top of the Makefile, e.g. `make train TRAIN_CONFIG=conf/experiments/wafer_augment.yaml SEED=1`.

| Stage | Command | Purpose |
|---|---|---|
| Setup | `make conda-dev` | Create (or reuse) the conda environment and initialize the project (**recommended for initial setup**); `ENV=...`, `TORCH=cpu`, `FORCE=1` |
| | `make init` | Install dependencies, the project and the hooks in the **currently active** environment (`TORCH=cpu` works here too) |
| Quality and security | `make lint`, `make code-format` | Run all pre-commit checks; format code and fix lint issues with Ruff |
| | `make typecheck` | Type-check `src` and `scripts` with mypy |
| | `make test` | Run the test suite with coverage |
| | `make audit` | Check the lock files against known vulnerabilities (pip-audit) |
| Dependencies | `make requirements` | Re-lock `requirements.txt` / `requirements-dev.txt` from the `.in` files |
| | `make requirements-check` | Compare the code's imports with `requirements.in`; read-only, it changes no file |
| Data | `make eda`, `make split` | Explore the dataset; create a fixed split |
| Training | `make sanity-check`, `make train`, `make train-seeds` | Overfit one batch; train one run; train over seeds |
| Experiments | `make compare`, `make benchmark`, `make search` | Compare configs over seeds; measure speed; multi-objective search |
| Final evaluation | `make evaluate-test`, `make gradcam`, `make export-onnx` | Test evaluation; Grad-CAM analysis; export and register the model |
| Tracking | `make mlflow-ui` | Open the local MLflow UI |
| Maintenance | `make clean` | Delete caches (`__pycache__`, `.pytest_cache`, ...), all safe to regenerate |

Some steps need results that are not in git. The commands that take a `RUN_ID` (`analyze-errors`,
`edge-coverage`, `gradcam`, `export-onnx`) read a run from your local MLflow store (`mlruns/` and its
database stay on the machine that trained): train first, then find the run in `make mlflow-ui`.
`make evaluate-test` is the project's single test evaluation and refuses to run twice; its results are
committed in `reports/final/`.


## Dependency Package Management

This project uses `pip-tools` to split dependency declarations into two levels:
- **`requirements.in` / `requirements-dev.in`**: Manually maintained lists of "direct" dependencies, listing only the packages actually used by the code.
- **`requirements.txt` / `requirements-dev.txt`**: Generated via `make requirements` after automatically resolving and locking versions; **do not edit manually**, and ensure both files are committed to version control.
When adding a new dependency:

1. Add it to the corresponding `.in` file (`make requirements-check` lists the imports that `requirements.in` does not declare).
2. Run `make requirements` to re-lock the versions.
3. Commit both the modified `.in` and `.txt` files.
For some packages, the import name differs from the PyPI package name (e.g., `import cv2` corresponds to `opencv-python-headless`), and import scanners guess these mappings; this is why `make requirements-check` only reports, and `requirements.in` is always edited by hand.

### Security

- **`constraints.txt`** holds security floors, minimum versions applied to every recompile, so the lock
  files can never fall back to a known vulnerable version. Each floor names its advisory.
- **`requirements-pip.txt`** installs pip itself from a URL pinned by SHA-256, safe even if the
  connection were intercepted (`make pip-bootstrap`; every install runs it first, in `shell/install_deps.sh`).
- **`make audit`** checks the lock files against known vulnerabilities; CI runs it on every push and
  every week.
- pip-compile keeps existing pins. To meet a new floor for packages that pin each other exactly, such
  as MLflow's or OpenTelemetry's, release them together: `make requirements UPGRADE="mlflow mlflow-skinny mlflow-tracing"`.


## Continuous Integration

Every push or pull request triggers two GitHub Actions workflows:

- **CI** (`.github/workflows/ci.yml`):
  - **pre-commit checks**: Uses the same configuration as the local `.pre-commit-config.yaml`, thereby avoiding discrepancies where local checks pass but CI fails.
  - **Test suite**: Installs the dependencies with the same script as the local setup (`shell/install_deps.sh`), with `TORCH=cpu` since the CI machine has no GPU, then executes pytest and generates a coverage report.
- **Dependency audit** (`.github/workflows/audit.yml`): checks the lock files against known
  vulnerabilities, also every Monday, since a lock file that does not change can become vulnerable
  when a new advisory is published.


## Data Quality Findings

Issues found in the official dataset, and how they are handled. The raw file is never modified:
every rule below is applied in code and configured under `conf/`.

- **Extra duplicate images.** 1,024 images are byte-for-byte copies of another image with the same labels.
  1,000 of them form a complete second copy of the Center+Edge_Loc+Scratch combination, which explains
  why the documentation lists 2,000 samples for that combination and 1,000 for every other one. The
  remaining 24 belong to Random (16) and Near_Full (8). A first random split left 157 validation and
  174 test images with an identical copy in the training split, inflating the validation scores.
  Split `v2` keeps a single copy of each image, and every reported result uses it.
- **Undocumented pixel values.** The documentation lists three pixel values (0 = empty, 1 = pass,
  2 = fail), but the file contains 214 pixels with the value 3 in 105 images, all labeled Random (97)
  or Near_Full (8): the two classes whose labels the authors corrected after release. They are imputed
  in memory with the most common valid value among their neighbors (194 become fail, 20 pass).


## Methodology

The full account, with the reasoning behind each design decision, is in
[docs/methodology.md](docs/methodology.md). In short:

- **A trustworthy evaluation first.** All decisions use the validation split; the test split is
  evaluated once, at the end. Runs train the full cosine schedule and keep the final epoch, and
  configurations are compared over 3 seeds. The earlier protocol, keeping the best epoch with early
  stopping, biases scores upward and had made run-to-run noise six times larger. The accuracy metric
  is the number of misclassified samples: macro average precision saturated at about 0.998 and could
  no longer rank models.
- **Understanding the errors before chasing them.** A rule-based measure of how far a defect wraps
  around the wafer edge showed that the most frequent confusion, Edge_Loc vs Edge_Ring, comes from
  labels that contradict the class definitions: every such error of the baseline lies in the top 4%
  of Edge_Loc coverage. Accuracy is limited by label quality, so the focus moved to efficiency.
- **Capacity is not the bottleneck.** Five architectures from 94 to 3,047 MFLOPs reach the same
  accuracy within noise, each at a learning rate chosen by the same pre-declared rule:
  | Architecture | MFLOPs | Validation errors (3 seeds) |
  |---|---|---|
  | EfficientNet-B0 | 646 | 42.1 ± 2.0 |
  | ShuffleNetV2 1.0x | 250 | 43.1 ± 2.1 |
  | MobileNetV3-Large | 355 | 43.9 ± 1.0 |
  | ResNet-18 | 3,047 | 44.1 ± 4.2 |
  | MobileNetV3-Small | 94 | 46.2 ± 0.2 |
- **Compute is not speed.** Benchmarks on GPU and CPU (ONNX Runtime) show that FLOPs predict CPU
  latency within an architecture family but not across families, where ResNet-style convolutions run
  about twice as many FLOPs per millisecond as mobile ones, and do not predict GPU latency, which
  follows the number of layers: ResNet-18, the most expensive model, is faster on the GPU than every
  mobile architecture.
- **Mapping the trade-off, then checking it.** A multi-objective Optuna search over slim ResNets
  (validation errors vs FLOPs, 50 screening trials) suggested accuracy stays flat down to about 33
  MFLOPs. Confirmation with the full protocol overturned this: the slim ResNets are the fastest models
  on every target, but less accurate than off-the-shelf architectures at similar compute, so none was
  eligible. A negative result, caught before it could drive the selection.
- **Selection rules fixed in advance.** How one model is chosen for each deployment target (cloud GPU,
  CPU server, edge) is written in [docs/model_selection.md](docs/model_selection.md), committed before
  the search and the test evaluation, so the choice cannot be adapted to the results.
- **Explaining the model, with a control.** Grad-CAM heatmaps of the selected model are checked against
  the same architecture with random weights, and the measurements were calibrated on synthetic wafers
  with known defect positions before measuring the real model. Edge_Loc heatmaps first seemed to point
  away from the defect; a diagnostic, with its possible outcomes written down in advance, showed why:
  they mark where the edge stops failing, the evidence that separates a partial edge defect from a ring.


## Results

**Selected model: ShuffleNetV2 1.0x**, trained from scratch with augmentation and a learning rate of
3e-3, for all three deployment targets (cloud GPU, CPU server, edge). It was chosen on the validation
split by the rules in [docs/model_selection.md](docs/model_selection.md), committed before the test
split was used (tag `selection-protocol`): among the candidates tied on accuracy, it has the highest
GPU throughput and the lowest CPU and edge latency.

### What the model looks at

Grad-CAM heatmaps of the delivered model on the validation split, each measurement compared with the
same architecture with random weights ([reports/final/gradcam.md](reports/final/gradcam.md);
[docs/methodology.md](docs/methodology.md), section 9):

- **The heatmaps reflect what the model learned**: they are nearly uncorrelated with those of a model
  with random weights (rank correlation 0.01 to 0.16).
- **Compact defects are localized**: the peak of every Center heatmap lies in the center of the wafer
  (8% for random weights), and in the randomly drawn examples, Scratch heatmaps follow the scratch
  line and Loc heatmaps cover the cluster.
- **Edge_Loc is recognized by where the edge stops failing.** The heatmaps avoid the failed part of the
  edge (0.34 times the mass its size would give) and concentrate where the failure ends and on the edge
  that did not fail (2.00 and 1.78; random weights: about 1 everywhere). Telling a partial edge defect
  from a ring is exactly a question of where the failure stops. It is consistent with the label
  confusion: an edge that fails almost all around offers almost no such place, and the Edge_Loc maps
  predicted as Edge_Ring are precisely those.

### Test results

The single evaluation on the held-out test split (3,700 samples). Mean ± standard deviation over 3
training seeds, with per-class thresholds tuned on the validation split. The 95% interval reflects the
finite size of the test set (bootstrap over test samples).

| Metric | ShuffleNetV2 1.0x |
|---|---|
| Misclassified samples (any label wrong) | 46.0 ± 1.7 of 3,700 (95% interval 34.3 to 58.7) |
| Exact match (all 8 labels right) | 98.76% ± 0.05% |
| Macro F1 | 0.9932 ± 0.0004 |
| Macro average precision | 0.9995 |
| Size | 1.26M parameters, 250 MFLOPs per image, 5.3 MB as ONNX |
| Speed | GPU 2.31 ms per image and 7,336 images/s in batches of 256; CPU server 0.82 ms; edge proxy 1.43 ms |

Speed measured on an RTX 5060 Ti (PyTorch FP32) and a Ryzen 5 9600X (ONNX Runtime, 6 threads for the
CPU server, 1 thread for the edge proxy); see [reports/experiments/benchmark.md](reports/experiments/benchmark.md).

Per class, on the test split (mean over seeds):

| Class | Test samples | Precision | Recall | F1 |
|---|---|---|---|---|
| Center | 1,200 | 1.0000 | 1.0000 | 1.0000 |
| Donut | 1,200 | 0.9997 | 1.0000 | 0.9999 |
| Edge_Loc | 1,200 | 0.9986 | 0.9894 | 0.9940 |
| Edge_Ring | 1,200 | 0.9947 | 0.9992 | 0.9970 |
| Loc | 1,800 | 0.9989 | 0.9943 | 0.9966 |
| Near_Full | 15 | 0.9375 | 1.0000 | 0.9677 |
| Scratch | 1,800 | 0.9974 | 0.9920 | 0.9947 |
| Random | 85 | 1.0000 | 0.9922 | 0.9961 |

### Every candidate on the test set

Evaluated with the same procedure for transparency; these results do not change the selection.
The last column is the selected model's errors minus the candidate's, on identical resamples of the
test set: an interval containing 0 means no detectable difference.

| Candidate | MFLOPs | Test errors | 95% interval | ShuffleNetV2 minus candidate |
|---|---|---|---|---|
| **ShuffleNetV2 1.0x (selected)** | 250 | 46.0 ± 1.7 | 34.3 to 58.7 | |
| EfficientNet-B0 | 646 | 42.7 ± 1.5 | 32.7 to 54.3 | +3.3 (−3.3 to +9.7) |
| MobileNetV3-Large | 355 | 45.3 ± 4.0 | 34.3 to 57.3 | +0.7 (−6.0 to +7.7) |
| ResNet-18 | 3,047 | 48.7 ± 3.1 | 37.0 to 61.3 | −2.7 (−9.3 to +4.3) |
| MobileNetV3-Small | 94 | 49.3 ± 8.3 | 38.3 to 61.7 | −3.3 (−11.3 to +4.7) |
| Slim ResNet, width 16, 2 blocks, small stem | 192 | 52.7 ± 1.5 | 40.7 to 66.0 | −6.7 (−13.3 to +0.3) |
| Slim ResNet, width 32, 1 block, small stem | 349 | 59.7 ± 7.1 | 46.7 to 73.3 | −13.7 (−21.3 to −6.3) |
| Slim ResNet, width 32, 1 block, standard stem | 33 | 61.0 ± 2.0 | 48.3 to 74.3 | −15.0 (−23.7 to −6.3) |
| Slim ResNet, width 24, 1 block, standard stem | 20 | 70.0 ± 6.1 | 56.3 to 85.0 | −24.0 (−34.3 to −14.3) |
| Slim ResNet, width 16, 2 blocks, standard stem | 18 | 76.3 ± 2.5 | 63.3 to 91.7 | −30.3 (−41.0 to −20.3) |

### What the results show

- **The validation conclusions hold on unseen data.** For every off-the-shelf candidate, test and
  validation errors at the default threshold differ by 1 to 2. The project took many decisions on the
  validation split (learning rates, a 50-trial search, the selection); none of them overfitted it.
- **No off-the-shelf architecture differs detectably from the selected model**, from 94 to 3,047
  MFLOPs, on the test set as on the validation set. Choosing among them by efficiency is justified.
- **The slim ResNets trade accuracy for speed.** They are the fastest models on every target, but four
  of the five are significantly less accurate than the selected model and the fifth nearly so.
- **The remaining errors sit where the labels are ambiguous.** Edge_Loc has the lowest recall and
  Edge_Ring the lowest precision: ring-like Edge_Loc samples are predicted as Edge_Ring, the label
  conflict found by the error analysis ([docs/methodology.md](docs/methodology.md)).

The deliverable is the seed-0 model of the selected configuration with its tuned thresholds, fixed in
advance rather than picked as the best seed. Full results, with the commit of the evaluated code:
[reports/final/test_evaluation.md](reports/final/test_evaluation.md).


## Using the delivered model

The delivered model is a single, self-contained ONNX file of 5.3 MB. It takes raw wafer maps and
returns one probability per defect class; the class names, the decision thresholds and the model's
provenance are in its metadata. Its [model card](reports/final/model_card.md) documents the
input and output contract, the performance and the limitations.

### Getting the model

- **From the GitHub release** [v1.0.0](../../releases/tag/v1.0.0): download
  `wafer_defect_classifier.onnx` and check that its SHA-256 matches the one in the model card:
```bash
  sha256sum wafer_defect_classifier.onnx
  # bc39e046c7d687fc70f5bc89d4778e16ba8478ffd11f19a4a690b0eddfea924d
```

- **From the MLflow model registry**, on the machine where it was trained, by alias rather than by
  file path:
```python
  import mlflow

  model_dir = mlflow.artifacts.download_artifacts("models:/wafer-defect-classifier@champion")
  # model_dir/model.onnx
```

  The alias `champion` points to the version in use. Moving it to a newer version updates every
  consumer without changing their code.

### Running inference

The reference implementation needs NumPy and ONNX Runtime only:

```python
import numpy as np

from src.inference.classifier import WaferDefectClassifier

classifier = WaferDefectClassifier("wafer_defect_classifier.onnx")

# uint8 maps shaped (N, 52, 52): 0 outside the wafer, 1 pass, 2 fail
maps = np.load("wafer_maps.npy")

labels = classifier.predict_labels(maps)  # e.g. [["Center", "Scratch"], []]
probabilities = classifier.predict_proba(maps)  # (N, 8), in the order of classifier.spec.class_names
```

- `predict_labels` and `predict` apply the per-class thresholds stored in the file; `predict_proba`
  returns the probabilities, for an application that needs its own thresholds.
- Pixel values out of range are handled as in training, imputed from their neighbors.
- A single map shaped (52, 52) is accepted as well.

### Without this repository

The file works with any ONNX Runtime binding (C++, C#, Java, JavaScript, ...):

- Input `wafer_map`: uint8, shape (N, 52, 52). Output `probabilities`: float32, shape (N, 8).
- The metadata holds `class_names` and `thresholds` as JSON lists, in the order of the outputs; a class
  is present when its probability reaches its threshold.
- `format_version` identifies the metadata layout: refuse a version you do not know.
`src/inference/classifier.py` is the executable specification of this contract.

### Reproducing the export

```bash
make export-onnx RUN_ID=176643c399f14d199631a45c01f3f427
```

It exports the model from its MLflow run, checks the ONNX file against PyTorch on the validation split
(largest probability difference 1.2e-6, no map with a different decision), registers it and writes
the model card. Running it again registers a new version with a new SHA-256, since the file records
the commit it was exported from; the delivered file is version 1.

The run id above is the delivered run in the original MLflow store, which is not part of the
repository. On a fresh clone, first train the configuration (with its overrides) and the deliverable
seed named in `conf/evaluation/final_test.yaml`, then pass the id of that run.


## Limitations

### Data

- **Partly synthetic data.** Many mixed-type maps in MixedWM38 were generated by a GAN. Results on maps
  from a real production line may differ.
- **Ambiguous labels between Edge_Loc and Edge_Ring.** About 3% of Edge_Loc samples cover nearly the
  whole wafer edge, where the labels contradict the class definitions; the most frequent errors come
  from this region. Accuracy there reflects how the data was labeled more than what the model learned.
- **Artificial class balance.** The dataset is balanced by design: most defect classes have 1,200 to
  1,800 test samples. On a production line, defective wafers are expected to be a minority, so
  precision in production would differ from these figures.
- **Rare classes are measured imprecisely.** Near_Full has 15 test samples and Random 85; a single
  error moves their per-class metrics by several points (Near_Full's precision of 0.9375 is about one
  false positive per seed). Near_Full and Random also never appear combined with other defects, so
  such combinations are untested.
- **Orientation invariance is assumed.** Training uses 90-degree rotations and flips, which assumes a
  defect class does not depend on the wafer's orientation. This holds for the 8 classes here, but would
  not for a defect defined relative to the notch or to equipment orientation.
- **Fixed input format.** Models expect 52x52 maps with values 0, 1 and 2; other wafer sizes need
  resizing or retraining.

### Evaluation

- **One split.** Every conclusion comes from a single train/validation/test split. Training randomness
  is measured (3 seeds); variation across splits is not.
- **A test set of 3,700 samples resolves about ±12 errors.** Differences of a few errors between
  models, on either split, cannot be told apart from noise; the paired comparisons above narrow this to
  about ±7.
- **Three seeds estimate variance roughly.** MobileNetV3-Small had a standard deviation of 0.6 errors
  across seeds on the validation split, but of 8.3 on the test split.
- **Per-class threshold tuning did not help.** Tuned thresholds increased test errors for all ten
  candidates, by 2.8 on average (by 0.3 for the selected model). Two likely reasons: thresholds were
  chosen to maximize per-class F1 while the primary metric counts samples with any wrong label, and with
  only 40 to 70 validation errors per model, each threshold followed a handful of samples. The thresholds
  varied most across seeds for Edge_Loc and Edge_Ring, the classes with ambiguous labels. As the
  protocol requires, the deliverable keeps its tuned thresholds; revisiting them needs new data, since
  the test split has now been used.

### Model selection

- **The eligibility rule was too coarse.** A candidate was considered as accurate as the best one if
  its mean validation errors were within 4, a margin based on training noise alone. It ignores the
  sampling noise of the validation set, yet simply widening it would have admitted slim ResNets that
  the test set shows to be less accurate. A paired comparison with the best candidate on the
  validation split, as done on the test split, would have been the principled criterion.
- **The selection is sensitive to that margin.** MobileNetV3-Small missed it by 0.13 errors. It is
  faster than the selected model on every target and statistically indistinguishable from it on the
  test set.
- **The learning-rate check was incomplete.** It only covered candidates that would be selected, so a
  candidate excluded because its learning rate did not suit it was never rechecked; MobileNetV3-Small
  had 2.6 fewer errors in a single run at a higher rate.
- **Tuning effort was unequal.** The slim ResNets were tuned in a 50-trial search; the off-the-shelf
  architectures over three learning rates each.
These flaws are reported rather than corrected: changing the rules after seeing the results is what
the pre-registered protocol exists to prevent.

### Efficiency and deployment

- **One machine, one runtime per target.** Speeds come from one GPU and one CPU, with PyTorch FP32 on
  the GPU and ONNX Runtime FP32 on the CPU. Deployment optimizations (TensorRT, reduced precision,
  quantization, dynamic batching) were not evaluated and may change the ranking, especially on the GPU.
- **Edge figures are a proxy.** They come from a single desktop CPU core, several times faster than
  the ARM cores of typical edge devices: they rank models, they do not predict latency on a device.
- **Model computation only.** GPU timings start with the input already on the GPU; preprocessing,
  data transfer and serving overhead are not included.
- **No production evaluation.** Behavior under data drift, on new defect types, or with real
  production prevalence is untested.

### Explainability

- **Coarse heatmaps.** Grad-CAM from the last convolutional block is 7x7: it resolves direction around
  the wafer but radius only to about one cell, and spreads patterns that extend along the whole edge,
  such as Edge_Ring and Random, into diffuse maps that localize nothing.
- **Mixed defects blur the attribution.** On maps with several defects, a class's heatmap may highlight
  another defect; heatmaps of the confused Edge_Loc samples do not show why each one was confused.
- **One assumed zone was wrong.** The Donut zone came from an assumed geometry, and the donuts appear
  to lie closer to the center than assumed, so its pointing result cannot be interpreted. Zones should be
  derived from the data.
- **One model, one split.** The analysis covers the delivered model (seed 0) on the validation split.
  It describes how this model decides; it does not show that other seeds or architectures decide the
  same way.


## Documentation

| Document | Contents |
|---|---|
| [docs/methodology.md](docs/methodology.md) | How the models were developed and evaluated, and why |
| [docs/model_selection.md](docs/model_selection.md) | The pre-registered selection rules, the selection and the test evaluation protocol |
| [reports/final/](reports/final/) | Test evaluation, Grad-CAM analysis and model card (generated) |
| [reports/experiments/](reports/experiments/) | Comparisons, benchmarks and search results (generated) |
| [docs/zh-TW/](docs/zh-TW/) | Traditional Chinese versions of the documents above |


## Project Structure

```
├── conf/                   # YAML configs: dataset, split, models, experiments, search, evaluation
├── data/
│   ├── raw/                # MixedWM38 (not in git, see Setup)
│   └── splits/             # the fixed split indices (v1, v2)
├── docs/                   # methodology and selection rules; zh-TW/ holds the Chinese versions
├── notebooks/              # exploratory notebook (outputs stripped by nbstripout)
├── reports/
│   ├── experiments/        # comparisons, benchmarks, search results (generated)
│   └── final/              # test evaluation, Grad-CAM, model card (generated)
├── scripts/                # command-line entry points, run through the Makefile
├── src/
│   ├── common/             # config schema, paths, hashing
│   ├── data/               # loading, cleaning, splitting, transforms
│   ├── eda/                # exploratory analysis and plots
│   ├── models/             # model factory, profiling, benchmarks, ONNX export
│   ├── training/           # training pipeline, metrics, MLflow tracking, search
│   ├── evaluation/         # error analysis, thresholds, bootstrap, Grad-CAM, model card
│   └── inference/          # reference inference, NumPy and ONNX Runtime only
├── tests/                  # pytest suite
├── shell/                  # environment setup (setup_env, install_deps, init_project) and dependency scripts
├── constraints.txt         # security floors for the lock files
├── requirements*.in / .txt # direct dependencies and lock files
├── requirements-pip.txt    # pip, pinned by hash
└── Makefile                # every command: run `make` to list them
```


## License

This project is licensed under the [Apache License 2.0](LICENSE). The MixedWM38 dataset is not part of
this repository and remains subject to its authors' terms.
