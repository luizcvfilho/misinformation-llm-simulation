.PHONY: help setup sync sync-dev lock add notebook precommit-install precommit-run notebooks notebooks-inplace notebooks-continue fetch-news clean

ifeq ($(OS),Windows_NT)
VENV_DIR ?= .venv
HELP_CMD := findstr /R "^[a-zA-Z_-][a-zA-Z0-9_-]*:.*##" Makefile
CLEAN_CMD := powershell -NoProfile -Command "if (Test-Path $(VENV_DIR)) { Remove-Item -Recurse -Force $(VENV_DIR) }"
else
VENV_DIR ?= .venv-linux
HELP_CMD := grep -E "^[a-zA-Z_-][a-zA-Z0-9_-]*:.*##" Makefile
CLEAN_CMD := rm -rf "$(VENV_DIR)"
endif

export UV_PROJECT_ENVIRONMENT := $(VENV_DIR)

NOTEBOOKS ?= src/llm_simulation_workbench.ipynb src/bert_fake_real_workbench.ipynb
OUTPUT ?= data/newsdata_news.csv
LANGUAGE ?= en
COUNTRY ?=
CATEGORY ?=
QUERY ?=
MAX_RECORDS ?= 200

help: ## List available targets
	@$(HELP_CMD)

setup: ## Create .venv, generate lockfile, and sync dependencies (including dev)
	uv venv
	uv lock
	uv sync --group dev

sync: ## Sync environment with pyproject.toml and uv.lock
	uv sync

sync-dev: ## Sync environment including development dependencies
	uv sync --group dev

lock: ## Update uv.lock from pyproject.toml
	uv lock

add: ## Add dependency (usage: make add PKG=package-name)
	uv add $(PKG)

notebook: ## Open Jupyter Lab using uv
	uv run jupyter lab

precommit-install: ## Install pre-commit hooks in the local repository
	uv run pre-commit install

precommit-run: ## Run all hooks across the project
	uv run pre-commit run --all-files

notebooks: ## Run notebooks sequentially and save to output/executed_notebooks
	uv run python src/run_notebooks_sequentially.py --notebooks $(NOTEBOOKS)

notebooks-inplace: ## Run notebooks sequentially and save in-place
	uv run python src/run_notebooks_sequentially.py --notebooks $(NOTEBOOKS) --inplace

notebooks-continue: ## Run notebooks and continue even if one fails
	uv run python src/run_notebooks_sequentially.py --notebooks $(NOTEBOOKS) --continue-on-error

fetch-news: ## Fetch news from NewsData.io and save as CSV
	uv run python src/fetch_newsdata_to_csv.py --output $(OUTPUT) --language $(LANGUAGE) --country $(COUNTRY) --category $(CATEGORY) --query $(QUERY) --max-records $(MAX_RECORDS)

clean: ## Remove virtual environment
	@$(CLEAN_CMD)
