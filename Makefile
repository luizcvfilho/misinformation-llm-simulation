.PHONY: help setup sync sync-dev lock add notebook precommit-install precommit-run test coverage coverage-html lint format lint-format notebooks notebooks-inplace notebooks-continue fetch-news audit-topic-domain-coverage prepare-stdi-manual-evaluation calibrate-stdi topic-drift-comparison stdi-logistic-regression csv-explorer interaction-graph interaction-graph-verbose interaction-graph-ui clean

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
DOMAIN_AUDIT_INPUT ?= data/raw/newsdata_news.csv
DOMAIN_AUDIT_OUTPUT_DIR ?= output/audit/topic_domain_coverage
DOMAIN_AUDIT_MAX_ROWS ?= 120
DOMAIN_AUDIT_MAX_REQUESTS_PER_MINUTE ?= 60
DOMAIN_AUDIT_MODEL ?=
DOMAIN_AUDIT_PROVIDER ?=
STDI_MANUAL_OUTPUT_DIR ?= output/stdi_manual_evaluation
STDI_MANUAL_INPUT ?= data/raw/newsdata_news.csv
STDI_MANUAL_SAMPLE_SIZE ?= 50
STDI_MANUAL_MODEL ?=
STDI_MANUAL_PROVIDER ?=
STDI_MANUAL_MAX_REQUESTS_PER_MINUTE ?= 450
STDI_MANUAL_GENERATE ?=
STDI_MANUAL_SCORE ?=
STDI_MANUAL_WITHOUT_VAD ?=
STDI_REGRESSION_INPUT ?=
STDI_REGRESSION_OUTPUT_DIR ?= output/stdi_logistic_regression
STDI_REGRESSION_TRUE_REFERENCE_INPUT ?=
STDI_REGRESSION_TEXT_COLUMN ?= text
STDI_REGRESSION_TRUE_TEXT_COLUMN ?=
STDI_REGRESSION_TITLE_COLUMN ?= title
STDI_REGRESSION_TOPIC_COLUMN ?= subject
STDI_REGRESSION_MODEL ?= gpt-5.6-luna
STDI_REGRESSION_PROVIDER ?= chatgpt
STDI_REGRESSION_MAX_ROWS ?=
STDI_REGRESSION_MAX_REQUESTS_PER_MINUTE ?= 450
STDI_REGRESSION_SKIP_REWRITE ?=
STDI_REGRESSION_TFIDF_MAX_FEATURES ?= 10000
STDI_REGRESSION_TFIDF_MIN_DF ?= 5
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
GRAPH_TOPIC_DRIFT_MODEL ?=
GRAPH_TOPIC_DRIFT_PROVIDER ?=
GRAPH_OUTPUT_DIR ?= output/interaction_graph
GRAPH_OUTPUT_PREFIX ?= simulation

FETCH_NEWS_OPTIONAL_ARGS := \
	$(if $(strip $(OUTPUT)),--output $(OUTPUT),) \
	$(if $(strip $(COUNTRY)),--country $(COUNTRY),) \
	$(if $(strip $(CATEGORY)),--category $(CATEGORY),) \
	$(if $(strip $(QUERY)),--query $(QUERY),)

DOMAIN_AUDIT_OPTIONAL_ARGS := \
	$(if $(strip $(DOMAIN_AUDIT_MODEL)),--model $(DOMAIN_AUDIT_MODEL),) \
	$(if $(strip $(DOMAIN_AUDIT_PROVIDER)),--provider $(DOMAIN_AUDIT_PROVIDER),)

INTERACTION_GRAPH_OPTIONAL_ARGS := \
	$(if $(strip $(GRAPH_NEWS_ID_COLUMN)),--news-id-column $(GRAPH_NEWS_ID_COLUMN),) \
	$(if $(strip $(GRAPH_MAX_ROWS)),--max-rows $(GRAPH_MAX_ROWS),) \
	$(if $(strip $(GRAPH_MAX_REQUESTS_PER_MINUTE)),--max-requests-per-minute $(GRAPH_MAX_REQUESTS_PER_MINUTE),) \
	$(if $(strip $(GRAPH_TOPIC_DRIFT_MODEL)),--topic-drift-model $(GRAPH_TOPIC_DRIFT_MODEL),) \
	$(if $(strip $(GRAPH_TOPIC_DRIFT_PROVIDER)),--topic-drift-provider $(GRAPH_TOPIC_DRIFT_PROVIDER),) \
	$(if $(strip $(GRAPH_ALLOW_TITLE_FALLBACK)),--allow-title-fallback,)

STDI_MANUAL_OPTIONAL_ARGS := \
	$(if $(strip $(STDI_MANUAL_MODEL)),--model $(STDI_MANUAL_MODEL),) \
	$(if $(strip $(STDI_MANUAL_PROVIDER)),--provider $(STDI_MANUAL_PROVIDER),) \
	$(if $(strip $(STDI_MANUAL_GENERATE)),--generate,) \
	$(if $(strip $(STDI_MANUAL_SCORE)),--score,) \
	$(if $(strip $(STDI_MANUAL_WITHOUT_VAD)),--without-vad,)

STDI_REGRESSION_OPTIONAL_ARGS := \
	$(if $(strip $(STDI_REGRESSION_TRUE_REFERENCE_INPUT)),--true-input $(STDI_REGRESSION_TRUE_REFERENCE_INPUT),) \
	$(if $(strip $(STDI_REGRESSION_TRUE_TEXT_COLUMN)),--true-text-column $(STDI_REGRESSION_TRUE_TEXT_COLUMN),) \
	$(if $(strip $(STDI_REGRESSION_MAX_ROWS)),--max-rows $(STDI_REGRESSION_MAX_ROWS),) \
	$(if $(strip $(STDI_REGRESSION_SKIP_REWRITE)),--skip-rewrite,)

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
	@echo "  test               Run the test suite"
	@echo "  coverage           Run tests and print coverage report"
	@echo "  coverage-html      Run tests and generate htmlcov/index.html"
	@echo "  lint               Run Ruff lint checks"
	@echo "  format             Format code with Ruff"
	@echo "  lint-format        Run Ruff lint and format in sequence"
	@echo "  notebooks          Run notebooks sequentially and save to output/runs/<run_id>"
	@echo "  notebooks-inplace  Run notebooks sequentially and save in-place"
	@echo "  notebooks-continue Run notebooks and continue even if one fails"
	@echo "  fetch-news         Fetch news from NewsData.io and save as CSV"
	@echo "  audit-topic-domain-coverage  Audit controlled-domain coverage using domain-only extraction"
	@echo "  prepare-stdi-manual-evaluation  Prepare or score 50 manually reviewed STDI pairs"
	@echo "  calibrate-stdi     Fit STDI weights from manual annotations"
	@echo "  topic-drift-comparison Run LLM or cluster comparison over shared structures"
	@echo "  stdi-logistic-regression Analyze STDI features using false/true reference groups"
	@echo "  csv-explorer       Open the generic CSV explorer"
	@echo "  interaction-graph  Run the interaction graph simulation"
	@echo "  interaction-graph-verbose Run the interaction graph simulation with progress logs"
	@echo "  interaction-graph-ui Open the Streamlit UI for the interaction graph workflow"
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

test: ## Run the test suite
	uv run pytest

coverage: ## Run tests and print coverage report
	uv run coverage run -m pytest
	uv run coverage report

coverage-html: ## Run tests and generate htmlcov/index.html
	uv run coverage run -m pytest
	uv run coverage html

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

audit-topic-domain-coverage: ## Audit domain coverage with a stratified sample
	uv run python scripts/audit_topic_domain_coverage.py --input $(DOMAIN_AUDIT_INPUT) --output-dir $(DOMAIN_AUDIT_OUTPUT_DIR) --max-rows $(DOMAIN_AUDIT_MAX_ROWS) --max-requests-per-minute $(DOMAIN_AUDIT_MAX_REQUESTS_PER_MINUTE) $(DOMAIN_AUDIT_OPTIONAL_ARGS)

prepare-stdi-manual-evaluation: ## Prepare or score 50 manually reviewed STDI pairs
	uv run python scripts/prepare_stdi_manual_evaluation.py --input $(STDI_MANUAL_INPUT) --output-dir $(STDI_MANUAL_OUTPUT_DIR) --sample-size $(STDI_MANUAL_SAMPLE_SIZE) --max-requests-per-minute $(STDI_MANUAL_MAX_REQUESTS_PER_MINUTE) $(STDI_MANUAL_OPTIONAL_ARGS)

calibrate-stdi: ## Fit STDI weights from manual annotations
	uv run python scripts/prepare_stdi_manual_evaluation.py --output-dir $(STDI_MANUAL_OUTPUT_DIR) --fit

topic-drift-comparison: ## Run a topic-drift comparison (set TOPIC_DRIFT_COMPARISON_ARGS)
	uv run python scripts/run_topic_drift_comparison.py $(TOPIC_DRIFT_COMPARISON_ARGS)

stdi-logistic-regression: ## Requires STDI_REGRESSION_INPUT; optional independent true-reference CSV
	uv run python scripts/run_stdi_logistic_regression.py --input $(STDI_REGRESSION_INPUT) --output-dir $(STDI_REGRESSION_OUTPUT_DIR) --text-column $(STDI_REGRESSION_TEXT_COLUMN) --title-column $(STDI_REGRESSION_TITLE_COLUMN) --topic-column $(STDI_REGRESSION_TOPIC_COLUMN) --provider $(STDI_REGRESSION_PROVIDER) --model $(STDI_REGRESSION_MODEL) --max-requests-per-minute $(STDI_REGRESSION_MAX_REQUESTS_PER_MINUTE) --tfidf-max-features $(STDI_REGRESSION_TFIDF_MAX_FEATURES) --tfidf-min-df $(STDI_REGRESSION_TFIDF_MIN_DF) $(STDI_REGRESSION_OPTIONAL_ARGS)

csv-explorer: ## Open the generic CSV explorer
	uv run streamlit run src/misinformation_simulation/apps/csv_explorer_app.py

interaction-graph:
	uv run python scripts/run_interaction_graph.py --input $(GRAPH_INPUT) --graph-config $(GRAPH_CONFIG) --text-column $(GRAPH_TEXT_COLUMN) --title-column $(GRAPH_TITLE_COLUMN) --sleep-seconds $(GRAPH_SLEEP_SECONDS) --retry-attempts $(GRAPH_RETRY_ATTEMPTS) --output-dir $(GRAPH_OUTPUT_DIR) --output-prefix $(GRAPH_OUTPUT_PREFIX) $(INTERACTION_GRAPH_OPTIONAL_ARGS)

interaction-graph-verbose:
	uv run python scripts/run_interaction_graph.py --input $(GRAPH_INPUT) --graph-config $(GRAPH_CONFIG) --text-column $(GRAPH_TEXT_COLUMN) --title-column $(GRAPH_TITLE_COLUMN) --sleep-seconds $(GRAPH_SLEEP_SECONDS) --retry-attempts $(GRAPH_RETRY_ATTEMPTS) --output-dir $(GRAPH_OUTPUT_DIR) --output-prefix $(GRAPH_OUTPUT_PREFIX) --verbose $(INTERACTION_GRAPH_OPTIONAL_ARGS)

interaction-graph-ui:
	uv run streamlit run src/misinformation_simulation/apps/interaction_graph_app.py

clean: ## Remove virtual environment
	@$(CLEAN_CMD)
