# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Scope: this file records what changes for someone *using* the repository.
Experiment runs and their metrics live in MLflow; the reasoning behind modelling
decisions lives in `docs/`.

繁體中文版：[CHANGELOG.zh-TW.md](CHANGELOG.zh-TW.md)

## [Unreleased]

## [1.0.2] - 2026-11-08

Remediation of the first findings raised by the scheduled dependency audit added
in 1.0.0, plus the guard rails that the remediation itself showed were missing.

### Security

- Raise the floor for `hydra-core` to `>= 1.3.8`, fixing GHSA-wf6j-7hvr-gr93
  (CVSS 3.1 7.8, High). The legacy target blocklist did not block PyYAML's unsafe
  loading functions, so untrusted configuration reaching the legacy
  `instantiate()` / `call()` path could execute arbitrary code. This project loads
  only first-party configuration, so real exposure was low.
- Raise the floor for `pyasn1` to `>= 0.6.5`, fixing CVE-2026-119194, -119195,
  -119196 and -119197 (CVSS 3.1 7.5 each, High). Quadratic-complexity denial of
  service in the BER/CER/DER decoders. Transitive via `google-auth`; the training
  pipeline does not decode externally supplied ASN.1.
- Both are recorded as floors in `constraints.txt` rather than left to the
  resolver, so a later recompile cannot silently regress to a vulnerable version.

### Added

- `tools/check_requirements.py`: reports every unparseable line in a requirements
  file in a single pass, and checks that a package pinned in more than one lock
  file agrees on the version. `pip-audit` and `pip` stop at the first bad line, so
  a corrupted lock otherwise costs one CI run per bad line.
- `make check-env`: fails unless the active conda environment matches `ENV`.

### Changed

- **`make requirements`, `test`, `sanity-check`, `train` and `export-onnx` now
  refuse to run outside the project's conda environment.** `conda-dev` is exempt
  because it creates that environment, and `audit` is exempt because it only reads
  the lock files.
- `pip` pinned to 26.3 (vendored certifi 2026.10.23, truststore 0.10.5), up from
  26.2.2 (vendored certifi 2026.10.19). No advisory applies; the bump keeps the
  vendored trust store current, which is the reason this pin exists.
- CI: `actions/checkout` v4 -> v5 and `actions/setup-python` v5 -> v6. Both still
  declared `node20`; GitHub removed Node 20 from Actions runners on 2026-09-23, so
  they were being force-run on Node 24 with a deprecation warning. v5 and v6 are
  the first release of each action to target Node 24.
- CI: `runs-on: ubuntu-latest` -> `ubuntu-26.04`. GitHub began migrating the
  `ubuntu-latest` label to Ubuntu 26.04 on 2026-10-19, completing by 2026-11-19,
  because the release reached general availability for production workloads — a
  routine image lifecycle change, not a security response. Pinning explicitly
  keeps CI reproducible and stops the label's meaning shifting mid-migration.
- Coverage omits `tests/` and `tools/`, which are tooling rather than product code.

## [1.0.1] - 2026-10-26

### Added

- A setup guide for a fresh clone.

### Changed

- One conda setup path, shared by local setup and CI, so `make conda-dev` and the
  CI job exercise the same script. Installing `src` is part of environment
  initialization, so a fresh clone can build the environment and run every step.
- Both READMEs completed and corrected for publication.
- Repository settings tidied.

## [1.0.0] - 2026-10-24

Initial release: the complete model development project, from dataset analysis to
a delivered ONNX model, using the Mixed-type Wafer Map Defect dataset.

### Added

- **Dataset EDA**: NPZ analysis, image and CSV packing, a general image reader,
  and a report of the pixel-value distribution against the documented range, with
  a configurable policy for undocumented pixel values.
- **Data pipeline**: config-driven fixed train/val/test splits, with sample
  exclusion and exact-duplicate removal (split v2). File hashing is shared between
  splitting and training, so every run records its data lineage.
- **Configuration**: typed schemas for models and training, validated against the
  class count.
- **Models**: a factory covering ResNet with a small-input stem, MobileNetV3,
  EfficientNet-B0 and ShuffleNetV2, each with a consistent small-input adaptation,
  plus a family of slim ResNets. Every model is built from its config in one place.
- **Training**: multi-label BCE with computed `pos_weight`, threshold-free macro
  AP, early stopping, learning-rate warmup, a final-epoch protocol, a
  single-batch sanity check, and MLflow tracking anchored to the project root.
  Seed and learning rate can be overridden per run.
- **Hyperparameter search**: a resumable multi-objective search over slim ResNets
  with Optuna (validation error against FLOPs), and export of selected trials as
  configs for confirmation runs.
- **Benchmarking**: model size, latency and throughput measured for the three
  deployment targets (cloud GPU, CPU server, edge).
- **Evaluation**: per-seed comparison of experiment configs from MLflow runs;
  validation error analysis with per-sample probabilities and error confidence,
  attached to the run; a rule-based Edge_Loc / Edge_Ring edge-coverage diagnostic;
  threshold tuning; bootstrap confidence intervals; and a single evaluation on the
  test split that refuses to run twice.
- **Explainability**: Grad-CAM analysis with localization, direction and
  randomization checks, plus an edge-profile diagnostic for Edge_Loc.
- **Delivery**: export, verification and registration of the selected model as a
  self-contained ONNX file, with a generated model card. ShuffleNetV2 1.0x was
  selected for all three deployment targets; see `docs/model_selection.md` for the
  protocol and the results.
- **Dependency management**: hand-curated `.in` files compiled to two lock files
  with pip-compile, where `requirements.txt` is constrained by
  `requirements-dev.txt` so both pin the same versions. `make requirements`
  recompiles; `UPGRADE="pkg ..."` releases chosen pins.
- **Documentation**: the model selection protocol, search design, methodology,
  results and limitations — including the Edge_Loc / Edge_Ring label ambiguity —
  with a Traditional Chinese version of every document.
- Apache 2.0 license.

### Security

- A dependency audit in CI: `pip-audit` against the lock files, weekly and on
  every push.
- `pip` installed from a hash-pinned direct URL, so the bootstrap is verifiable
  even if the TLS connection were intercepted. Added in response to the eMudhra /
  emSign CA certificate incident, which older pip trusts through its vendored
  certifi whatever the system trust store says.
- Security floors recorded in `constraints.txt` for packages with known
  vulnerabilities found by the audit during development, including `certifi` and
  `virtualenv`, and `filelock` and `mlflow` for advisories published while the
  release was under review.

[Unreleased]: https://github.com/poloH23/wafer-defect-model-dev/compare/v1.0.2...HEAD
[1.0.2]: https://github.com/poloH23/wafer-defect-model-dev/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/poloH23/wafer-defect-model-dev/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/poloH23/wafer-defect-model-dev/releases/tag/v1.0.0
