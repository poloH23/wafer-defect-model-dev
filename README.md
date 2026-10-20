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
- **Mapping the trade-off.** A multi-objective Optuna search over slim ResNets (validation errors vs
  FLOPs, 50 screening trials) finds accuracy flat down to about 33 MFLOPs, a third of
  MobileNetV3-Small, and dropping steeply below 20 MFLOPs. Its finalists are being confirmed with the
  full protocol.
- **Selection rules fixed in advance.** How one model is chosen for each deployment target (cloud GPU,
  CPU server, edge) is written in [docs/model_selection.md](docs/model_selection.md), committed before
  the search and the test evaluation, so the choice cannot be adapted to the results.



Readme results limitations · MD
## Results

**Selected model: ShuffleNetV2 1.0x**, trained from scratch with augmentation and a learning rate of
3e-3, for all three deployment targets (cloud GPU, CPU server, edge). It was chosen on the validation
split by the rules in [docs/model_selection.md](docs/model_selection.md), committed before the test
split was used (tag `selection-protocol`): among the candidates tied on accuracy, it has the highest
GPU throughput and the lowest CPU and edge latency. One model serves all three targets.

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


## Project Structure

<!-- TODO:  -->

## License

This project is licensed under the [Apache License 2.0](LICENSE)
