.PHONY: help code-format lint git-init-hook test \
        init requirements requirements-bootstrap requirements-refresh \
        conda-dev conda-export conda-custom \
        clean gitkeep-clean clean-zone \
        eda split sanity-check train train-seed lr-grid compare mlflow-ui test analyze-errors \
        docker-build docker-run docker-run-fastapi docker-run-flask \
        demo-run gitkeep-add system-info

# === 🧹 Cleaning ===
clean:  ## Clean the caches in the project
	rm -rf __pycache__ logs/* .pytest_cache .ruff_cache

clean-zone:  ## Remove Windows Zone.Identifier files
	find . -type f -name "*Zone.Identifier" -delete
	@echo "> Removed all .Zone.Identifier files in Linux."

# === 🧼 Code Formatting & Hooks ===
code-format:  ## Format code with ruff (lint fixes and formatting)
	ruff check . --fix
	ruff format .

lint:  ## Run all pre-commit hooks (ruff, gitkeep, etc)
	pre-commit run --all-files --show-diff-on-failure

git-init-hook:  ## Initialize pre-commit hooks to enable Git auto check
	pre-commit install
	@echo "> pre-commit hooks initialized"

# === 🚀 Initialization ===
init:  ## One-shot project initialization: install deps + pre-commit hook + baseline check
	chmod +x shell/*.sh
	bash shell/init_project.sh

# === 📦 Dependencies ===
requirements:  ## Recompile requirements.txt & requirements-dev.txt from .in files (pip-compile)
	bash shell/manage_requirements.sh compile

requirements-bootstrap:  ## Rescan source code imports to refresh requirements.in (overwrites existing file)
	bash shell/manage_requirements.sh bootstrap

requirements-refresh:  ## Full refresh: rescan imports AND recompile lock files in one step
	bash shell/manage_requirements.sh refresh

# === 📁 Gitkeep Files ===
gitkeep-add:  ## Add .gitkeep files in to empty directories
	bash shell/gitkeep.sh -v

gitkeep-clean:  ## Remove all .gitkeep files
	bash shell/gitkeep.sh -c

# === 📊 Data ===
EDA_CONFIG ?= conf/eda/wafer_defect.yaml
SPLIT_CONFIG ?= conf/split/wafer_defect_v1.yaml

eda:  ## Run dataset EDA (EDA_CONFIG=conf/eda/wafer_defect.yaml)
	python scripts/run_eda.py --config $(EDA_CONFIG)

split:  ## Create a fixed train/val/test split (SPLIT_CONFIG=conf/split/wafer_defect_v1.yaml)
	python scripts/run_split.py --config $(SPLIT_CONFIG)

# === 🤖 Training ===
TRAIN_CONFIG ?= conf/experiments/wafer_baseline.yaml

sanity-check:  ## Overfit one batch to verify the training pipeline (TRAIN_CONFIG=...)
	python scripts/train.py --config $(TRAIN_CONFIG) --sanity-check

SEED ?=
LR ?=
LRS ?= 3.0e-4 1.0e-3 3.0e-3

train:  ## Train a model and record the run in MLflow (TRAIN_CONFIG=..., optional SEED=..., LR=...)
	python scripts/train.py --config $(TRAIN_CONFIG) $(if $(SEED),--seed $(SEED)) $(if $(LR),--learning-rate $(LR))

SEEDS ?= 0 1 2

train-seeds:  ## Train one config once per seed (TRAIN_CONFIG=..., SEEDS="0 1 2", optional LR=...)
	for seed in $(SEEDS); do python scripts/train.py --config $(TRAIN_CONFIG) --seed $$seed $(if $(LR),--learning-rate $(LR)) || exit 1; done

lr-grid:  ## One run with seed 0 per learning rate (TRAIN_CONFIG=..., LRS="3.0e-4 1.0e-3 3.0e-3")
	for lr in $(LRS); do python scripts/train.py --config $(TRAIN_CONFIG) --seed 0 --learning-rate $$lr || exit 1; done

compare:  ## Compare experiment configs over seeds (CONFIGS="a.yaml b.yaml", optional OUTPUT=...)
	python scripts/compare_runs.py --configs $(CONFIGS) $(if $(OUTPUT),--output $(OUTPUT))

analyze-errors:  ## Analyze validation errors of a run and attach them to it in MLflow (RUN_ID=...)
	python scripts/analyze_errors.py --run-id $(RUN_ID)

edge-coverage:  ## Edge_Loc vs Edge_Ring edge coverage analysis of a run (RUN_ID=...; run analyze-errors first)
	python scripts/analyze_edge_coverage.py --run-id $(RUN_ID)

mlflow-ui: ## Open the local MLflow UI at http://127.0.0.1:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db

# === 🧪 Testing ===
test:  ## Run pytest
	pytest --cov=. --cov-report=term-missing

# === 🚀 Run Demo ===
demo-run: MODE=python  ## Run example runner (MODE=python|fastapi|flask, NGROK=1)
demo-run:
	python scripts/demo/main_demo.py --mode $(MODE) $(if $(NGROK),--ngrok)

# === 🧬 Conda Environment ===
conda-export:  ## Export conda env to file
	conda env export --from-history | grep -v "^prefix: " > bak/environment.yml
	@echo "> Conda environment has been exported to ./bak/environment.yml (history-only, avoids pip/conda version clashes)"

conda-dev:  ## Create conda env from environment.yml
	bash shell/setup_env.sh

conda-custom: ENV=myenv  ## Create lite version environment (ENV=myenv)
conda-custom:
	conda create -y -n $(ENV) python=3.12
	conda run -n $(ENV) bash shell/init_project.sh
	@echo "> Conda environment $(ENV) created and initialized"

# === 🐳 Docker ===
IMAGE ?= py-base
CONTAINER ?= $(IMAGE)-container

docker-build:  ## Build Docker image (IMAGE=py-base)
	docker build -t $(IMAGE) .

docker-run:  ## Run Docker image (IMAGE=py-base)
	docker run -it --name $(CONTAINER) $(IMAGE)

docker-run-fastapi:  ## Run the Docker image and mount the ports required by FastAPI (IMAGE=py-base)
	#docker run -p 8000:8000 py-base uvicorn main.__init__:app --host 0.0.0.0 --port 8000
	docker run -d --name $(CONTAINER) -p 8000:8000 $(IMAGE) \
	uvicorn scripts.demo.main_demo:app --host 0.0.0.0 --port 8000

docker-run-flask:  ## Run the Docker image and mount the ports required by Flask (IMAGE=py-base)
	#docker run -p 5000:5000 py-base flask --app main/__init__.py run --host=0.0.0.0
	docker run -d --name $(CONTAINER) -p 5000:5000 $(IMAGE) \
	flask --app scripts.demo.main_demo.py run --host=0.0.0.0

# === 🧱 System ===
system-info: ##  Display system resource status (CPU, memory, disk, Docker)
	bash shell/system_info.sh

# === 📘 Help ===
help:  ## Show categorized help
	@awk ' \
	/^# ===/ { print $$0; next } \
	/^[a-zA-Z0-9_-]+:.*?[[:space:]]+##/ { \
	  match($$0, /^([a-zA-Z0-9_-]+):.*?##[[:space:]]*(.*)$$/, arr); \
	  printf "  \033[36m%-20s\033[0m %s\n", arr[1], arr[2]; \
	}' $(MAKEFILE_LIST)
