.PHONY: help code-format lint git-init-hook test \
        init requirements requirements-bootstrap \
        conda-dev export-conda \
        clean gitkeep-clean clean-zone \
        docker-build docker-run docker-run-fastapi docker-run-flask \
        demo-run gitkeep-add

# === 🧹 Cleaning ===
clean:  ## Clean the caches in the project
	rm -rf __pycache__ logs/* .pytest_cache .ruff_cache

clean-zone:  ## Remove Windows Zone.Identifier files
	find . -type f -name "*Zone.Identifier" -delete
	@echo "> Removed all .Zone.Identifier files."

# === 🧼 Code Formatting & Hooks ===
code-format:  ## Format code with ruff (link fixes and formatting)
	ruff check . --fix
	ruff format .

lint:  ## Run all pre-commit hooks (black, ruff, gitkeep, etc)
	python -m pre_commit run --all-files --show-diff-on-failure

git-init-hook:  ## Initialize pre-commit hooks to enable Git auto check
	pre-commit install
	@echo "> pre-commit hooks initialized"

# === 🚀 Initialization ===
init:  ## One-shot project initialization: install deps + pre-commit hook + baseline check
	chmod +x shell/init_project.sh
	bash shell/init_project.sh

# === 📦 Dependencies ===
requirements:  ## Recompile requirements.txt & requirements-dev.txt from .in files (pip-compile)
	@command -v pip-compile >/dev/null 2>&1 || { echo "> pip-compile not found, bootstrapping dev tools from requirements-dev.in first"; pip install -r requirements-dev.in; }
	pip-compile requirements.in -o requirements.txt
	pip-compile requirements-dev.in -o requirements-dev.txt
	@echo "> requirements.txt and requirements-dev.txt regenerated"

requirements-bootstrap:  ## Rescan source code imports to refresh requirements.in (overwrites existing file)
	@command -v pipreqs >/dev/null 2>&1 || { echo "> pipreqs not found, bootstrapping dev tools from requirements-dev.in first"; pip install -r requirements-dev.in; }
	pipreqs . --savepath requirements.in --force
	@echo "> requirements.in refreshed from current imports — review the diff before committing"

requirements-refresh:  ## Full refresh: rescan imports AND recompile lock files in one step
	$(MAKE) requirements-bootstrap
	$(MAKE) requirements

# === 📁 Gitkeep Files ===
gitkeep-add:  ## Add .gitkeep files in to empty directories
	chmod +x shell/gitkeep.sh
	bash shell/gitkeep.sh -v

gitkeep-clean:  ## Remove all .gitkeep files
	chmod +x shell/gitkeep.sh
	bash shell/gitkeep.sh -c

# === 🧪 Testing ===
test:  ## Run pytest
	pytest --cov=. --cov-report=term-missing

# === 🚀 Run Demo ===
demo-run: MODE=python  ## Run example runner (MODE=python|fastapi|flask, NGROK=1)
demo-run:
	python scripts/demo/main_demo.py --mode $(MODE) $(if $(NGROK),--ngrok)

# === 🧬 Conda Environment ===
conda-export:  ## Export conda env to file
	conda env export | grep -v "^prefix: " > bak/environment.yml
	@echo "> Conda environment has been exported to bak/environment.yml"

conda-dev:  ## Create conda env from environment.yml
	chmod +x shell/setup_env.sh
	bash shell/setup_env.sh

conda-custom: ENV=myenv  ## Create lite version environment (ENV=myenv)
conda-custom:
	conda create -y -n $(ENV) python=3.12
	conda run -n $(ENV) pip install -r requirements.txt
	conda run -n $(ENV) pip install -r requirements-dev.txt
	conda run -n $(ENV) make git-init-hook
	@echo "> Conda environment $(ENV) created and pre-commit initialized

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
