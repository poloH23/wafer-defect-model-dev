# Run 'make' or 'make help' to list the commands by section. Sections follow the project's workflow,
# from setting up the environment to delivering the model. Variables in the block below can be
# overridden on the command line, e.g. 'make train TRAIN_CONFIG=conf/experiments/wafer_augment.yaml SEED=1'.

.DEFAULT_GOAL := help

# ==================================================================================================
# Variables (override on the command line)
# ==================================================================================================

# Data
EDA_CONFIG ?= conf/eda/wafer_defect.yaml
SPLIT_CONFIG ?= conf/split/wafer_defect_v2.yaml

# Training
TRAIN_CONFIG ?= conf/experiments/wafer_baseline.yaml
SEED ?=
SEEDS ?= 0 1 2
LR ?=
LRS ?= 3.0e-4 1.0e-3 3.0e-3

# Experiments and analysis
CONFIGS ?=
OUTPUT ?=
RUN_ID ?=
BENCH_MODELS ?= resnet18 mobilenet_v3_small mobilenet_v3_large efficientnet_b0 shufflenet_v2_x1_0
SEARCH_CONFIG ?= conf/search/slim_resnet_v1.yaml
TRIALS ?=

# Environment and dependencies
# - ENV: conda environment name. Plain "=" rather than "?=", so that a shell variable that happens to be
#   called ENV (POSIX shells use one) cannot replace it; "make conda-dev ENV=my-env" still overrides it.
# - TORCH: PyTorch build, cuda (the lock file as it is) or cpu (CPU-only, a much smaller download)
# - FORCE: FORCE=1 makes conda-dev remove the environment and rebuild it
# - UPGRADE: packages whose pins "make requirements" releases, e.g. UPGRADE="mlflow mlflow-skinny"
ENV = wafer-defect-model-dev
TORCH ?= cuda
FORCE ?=
UPGRADE ?=

# === 📘 Help ===
.PHONY: help

help:  ## Show the commands by section
	@awk 'BEGIN { FS = ":.*## " } \
	  /^# === / { title = $$0; gsub(/^# === | ===$$/, "", title); printf "\n%s\n", title; next } \
	  /^[a-zA-Z0-9_-]+:.*## / { printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@printf "\nVariables and their defaults are listed at the top of the Makefile.\n"

# === 🚀 Setup ===
.PHONY: conda-dev init pip-bootstrap git-init-hook check-env

conda-dev:  ## Create (or reuse) the conda env and initialize the project in it (ENV=..., TORCH=cuda|cpu, FORCE=1)
	ENV_NAME="$(ENV)" TORCH="$(TORCH)" FORCE="$(FORCE)" bash shell/setup_env.sh

init:  ## Initialize the project in the active env: deps, editable install, pre-commit hook, baseline check (TORCH=cuda|cpu)
	TORCH="$(TORCH)" bash shell/init_project.sh

pip-bootstrap:  ## Install pip from its hash-pinned URL (requirements-pip.txt), safe even if TLS is intercepted
	python -m pip install -r requirements-pip.txt

git-init-hook:  ## Install the pre-commit hooks, so that every commit is checked
	pre-commit install
	@echo "> pre-commit hooks initialized"

check-env:  ## Verify the active conda env matches ENV
	@test "$${CONDA_DEFAULT_ENV}" = "$(ENV)" || { \
	  printf '!! active conda env: %s\n!! expected:         %s\n   run: conda activate %s\n' \
	    "$${CONDA_DEFAULT_ENV:-<none>}" "$(ENV)" "$(ENV)"; exit 1; }

# === 📦 Dependencies ===
.PHONY: requirements requirements-check

requirements: check-env  ## Recompile the lock files from the .in files (pip-compile); UPGRADE="pkg ..." releases those pins
	bash shell/manage_requirements.sh compile $(foreach p,$(UPGRADE),--upgrade-package $(p))

requirements-check:  ## Compare the code's imports with requirements.in (read-only: no file is changed)
	bash shell/manage_requirements.sh check

# === 🧼 Code quality, security and tests ===
.PHONY: code-format lint typecheck audit test

code-format:  ## Format code with ruff (lint fixes and formatting)
	ruff check . --fix
	ruff format .

lint:  ## Run all pre-commit hooks (ruff, nbstripout, ...)
	pre-commit run --all-files --show-diff-on-failure

typecheck:  ## Type-check src and scripts with mypy (not part of CI yet)
	mypy src scripts

audit:  ## Check the lock files against known vulnerabilities (pip-audit)
	pip-audit --no-deps --disable-pip -r requirements.txt -r requirements-dev.txt

test: check-env  ## Run the test suite with coverage
	pytest --cov=. --cov-report=term-missing

# === 📊 Data ===
.PHONY: eda split

eda:  ## Run the dataset EDA (EDA_CONFIG=...)
	python scripts/run_eda.py --config $(EDA_CONFIG)

split:  ## Create a fixed train/val/test split (SPLIT_CONFIG=...)
	python scripts/run_split.py --config $(SPLIT_CONFIG)

# === 🤖 Training ===
.PHONY: sanity-check train train-seeds lr-grid

sanity-check: check-env  ## Overfit one batch to verify the training pipeline (TRAIN_CONFIG=...)
	python scripts/train.py --config $(TRAIN_CONFIG) --sanity-check

train: check-env  ## Train a model and record the run in MLflow (TRAIN_CONFIG=..., optional SEED=..., LR=...)
	python scripts/train.py --config $(TRAIN_CONFIG) $(if $(SEED),--seed $(SEED)) $(if $(LR),--learning-rate $(LR))

train-seeds:  ## Train one config once per seed (TRAIN_CONFIG=..., SEEDS="0 1 2", optional LR=...)
	for seed in $(SEEDS); do python scripts/train.py --config $(TRAIN_CONFIG) --seed $$seed $(if $(LR),--learning-rate $(LR)) || exit 1; done

lr-grid:  ## One run with seed 0 per learning rate (TRAIN_CONFIG=..., LRS="3.0e-4 1.0e-3 3.0e-3")
	for lr in $(LRS); do python scripts/train.py --config $(TRAIN_CONFIG) --seed 0 --learning-rate $$lr || exit 1; done

# === 🔬 Experiments and analysis ===
.PHONY: mlflow-ui compare analyze-errors edge-coverage benchmark search export-trials

mlflow-ui:  ## Open the local MLflow UI at http://127.0.0.1:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db

compare:  ## Compare experiment configs over seeds (CONFIGS="a.yaml b.yaml", optional OUTPUT=...)
	python scripts/compare_runs.py --configs $(CONFIGS) $(if $(OUTPUT),--output $(OUTPUT))

analyze-errors:  ## Analyze the validation errors of a run and attach them to it in MLflow (RUN_ID=...)
	python scripts/analyze_errors.py --run-id $(RUN_ID)

edge-coverage:  ## Edge_Loc vs Edge_Ring edge coverage analysis of a run (RUN_ID=...; run analyze-errors first)
	python scripts/analyze_edge_coverage.py --run-id $(RUN_ID)

benchmark:  ## Measure size, latency and throughput of architectures on this machine (BENCH_MODELS="...")
	python scripts/benchmark_models.py --dataset wafer_defect --models $(BENCH_MODELS) --output reports/experiments/benchmark.md

search:  ## Multi-objective search (validation errors vs FLOPs) with Optuna; rerun to resume (SEARCH_CONFIG=...)
	python scripts/search.py --config $(SEARCH_CONFIG)

export-trials:  ## Export search trials as configs for confirmation runs (SEARCH_CONFIG=..., TRIALS="15 46 10")
	python scripts/export_trial.py --config $(SEARCH_CONFIG) --trials $(TRIALS)

# === 🏁 Final evaluation and delivery ===
.PHONY: evaluate-test gradcam export-onnx

evaluate-test:  ## The single evaluation on the test split (docs/model_selection.md); refuses to run twice
	python scripts/evaluate_test.py --config conf/evaluation/final_test.yaml

gradcam:  ## Grad-CAM analysis of a model on the validation split (RUN_ID=..., e.g. the deliverable)
	python scripts/explain_gradcam.py --run-id $(RUN_ID)

export-onnx: check-env  ## Export, verify and register the delivered model, and write its model card (RUN_ID=...)
	python scripts/export_onnx.py --run-id $(RUN_ID)

# === 🧹 Maintenance ===
.PHONY: clean clean-zone gitkeep-add gitkeep-clean system-info

clean:  ## Remove caches and logs
	rm -rf __pycache__ logs/* .pytest_cache .ruff_cache

clean-zone:  ## Remove Windows Zone.Identifier files
	find . -type f -name "*Zone.Identifier" -delete
	@echo "> Removed all .Zone.Identifier files in Linux."

gitkeep-add:  ## Add .gitkeep files to empty directories
	bash shell/gitkeep.sh -v

gitkeep-clean:  ## Remove all .gitkeep files
	bash shell/gitkeep.sh -c

system-info:  ## Display system resource status (CPU, memory, disk, Docker)
	bash shell/system_info.sh
