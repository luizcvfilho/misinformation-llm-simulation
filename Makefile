.PHONY: help setup sync sync-dev lock add notebook precommit-install precommit-run lint format lint-format notebooks notebooks-inplace notebooks-continue fetch-news interaction-graph interaction-graph-verbose clean

.DEFAULT_GOAL := help

ifeq ($(OS),Windows_NT)
VENV_DIR ?= .venv
CLEAN_CMD := powershell -NoProfile -Command "if (Test-Path $(VENV_DIR)) { Remove-Item -Recurse -Force $(VENV_DIR) }"
else
VENV_DIR ?= .venv-linux
CLEAN_CMD := rm -rf "$(VENV_DIR)"
endif

export UV_PROJECT_ENVIRONMENT := $(VENV_DIR)

NOTEBOOKS ?= notebooks/llm_simulation_workbench.ipynb notebooks/bert_fake_real_workbench.ipynb notebooks/topic_drift_audit_workbench.ipynb notebooks/pretrained_fake_news_detector_workbench.ipynb
OUTPUT ?=
LANGUAGE ?= en
COUNTRY ?=
CATEGORY ?=
QUERY ?=
MAX_RECORDS ?= 200
GRAPH_INPUT ?= data/graph_news.csv
GRAPH_CONFIG ?= data/graph_config.json
GRAPH_TEXT_COLUMN ?= description
GRAPH_TITLE_COLUMN ?= title
GRAPH_NEWS_ID_COLUMN ?=
GRAPH_MAX_ROWS ?=
GRAPH_SLEEP_SECONDS ?= 0
GRAPH_MAX_REQUESTS_PER_MINUTE ?=
GRAPH_RETRY_ATTEMPTS ?= 5
GRAPH_ALLOW_TITLE_FALLBACK ?=
GRAPH_TOPIC_DRIFT_MODEL ?= gemini-3.1-flash-lite-preview
GRAPH_TOPIC_DRIFT_PROVIDER ?= gemini
GRAPH_OUTPUT_DIR ?= output/interaction_graph
GRAPH_OUTPUT_PREFIX ?= simulation

FETCH_NEWS_OPTIONAL_ARGS := \
	$(if $(strip $(OUTPUT)),--output $(OUTPUT),) \
	$(if $(strip $(COUNTRY)),--country $(COUNTRY),) \
	$(if $(strip $(CATEGORY)),--category $(CATEGORY),) \
	$(if $(strip $(QUERY)),--query $(QUERY),)

INTERACTION_GRAPH_OPTIONAL_ARGS := \
	$(if $(strip $(GRAPH_NEWS_ID_COLUMN)),--news-id-column $(GRAPH_NEWS_ID_COLUMN),) \
	$(if $(strip $(GRAPH_MAX_ROWS)),--max-rows $(GRAPH_MAX_ROWS),) \
	$(if $(strip $(GRAPH_MAX_REQUESTS_PER_MINUTE)),--max-requests-per-minute $(GRAPH_MAX_REQUESTS_PER_MINUTE),) \
	$(if $(strip $(GRAPH_ALLOW_TITLE_FALLBACK)),--allow-title-fallback,)

help: ## List available targets
	@echo "Available targets:"
	@echo "  help               List available targets"
	@echo "  setup              Create .venv, generate lockfile, and sync dependencies (including dev)"
	@echo "  sync               Sync environment with pyproject.toml and uv.lock"
	@echo "  sync-dev           Sync environment including development dependencies"
	@echo "  lock               Update uv.lock from pyproject.toml"
	@echo "  add                Add dependency (usage: make add PKG=package-name)"
	@echo "  notebook           Open Jupyter Lab using uv"
	@echo "  precommit-install  Install pre-commit hooks in the local repository"
	@echo "  precommit-run      Run all hooks across the project"
	@echo "  lint               Run Ruff lint checks"
	@echo "  format             Format code with Ruff"
	@echo "  lint-format        Run Ruff lint and format in sequence"
	@echo "  notebooks          Run notebooks sequentially and save to output/runs/<run_id>"
	@echo "  notebooks-inplace  Run notebooks sequentially and save in-place"
	@echo "  notebooks-continue Run notebooks and continue even if one fails"
	@echo "  fetch-news         Fetch news from NewsData.io and save as CSV"
	@echo "  interaction-graph  Run the interaction graph simulation"
	@echo "  interaction-graph-verbose Run the interaction graph simulation with progress logs"
	@echo "  clean              Remove virtual environment"

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

lint: ## Run Ruff lint checks
	uv run ruff check .

format: ## Format code with Ruff
	uv run ruff format .

lint-format: ## Run Ruff lint and format in sequence
	$(MAKE) lint
	$(MAKE) format

notebooks: ## Run notebooks sequentially and save to output/runs/<run_id>
	uv run python scripts/run_notebooks.py --notebooks $(NOTEBOOKS)

notebooks-inplace: ## Run notebooks sequentially and save in-place
	uv run python scripts/run_notebooks.py --notebooks $(NOTEBOOKS) --inplace

notebooks-continue: ## Run notebooks and continue even if one fails
	uv run python scripts/run_notebooks.py --notebooks $(NOTEBOOKS) --continue-on-error

fetch-news: ## Fetch news from NewsData.io and save as CSV
	uv run python scripts/fetch_newsdata.py --language $(LANGUAGE) $(FETCH_NEWS_OPTIONAL_ARGS) --max-records $(MAX_RECORDS)

interaction-graph:
	uv run python scripts/run_interaction_graph.py --input $(GRAPH_INPUT) --graph-config $(GRAPH_CONFIG) --text-column $(GRAPH_TEXT_COLUMN) --title-column $(GRAPH_TITLE_COLUMN) --sleep-seconds $(GRAPH_SLEEP_SECONDS) --retry-attempts $(GRAPH_RETRY_ATTEMPTS) --topic-drift-model $(GRAPH_TOPIC_DRIFT_MODEL) --topic-drift-provider $(GRAPH_TOPIC_DRIFT_PROVIDER) --output-dir $(GRAPH_OUTPUT_DIR) --output-prefix $(GRAPH_OUTPUT_PREFIX) $(INTERACTION_GRAPH_OPTIONAL_ARGS)

interaction-graph-verbose:
	uv run python scripts/run_interaction_graph.py --input $(GRAPH_INPUT) --graph-config $(GRAPH_CONFIG) --text-column $(GRAPH_TEXT_COLUMN) --title-column $(GRAPH_TITLE_COLUMN) --sleep-seconds $(GRAPH_SLEEP_SECONDS) --retry-attempts $(GRAPH_RETRY_ATTEMPTS) --topic-drift-model $(GRAPH_TOPIC_DRIFT_MODEL) --topic-drift-provider $(GRAPH_TOPIC_DRIFT_PROVIDER) --output-dir $(GRAPH_OUTPUT_DIR) --output-prefix $(GRAPH_OUTPUT_PREFIX) --verbose $(INTERACTION_GRAPH_OPTIONAL_ARGS)

clean: ## Remove virtual environment
	@$(CLEAN_CMD)
