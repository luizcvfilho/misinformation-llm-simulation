# LLM Misinformation Simulation Workbench

An LLM-based misinformation simulation framework with a rewrite, export, and factual-audit pipeline.

## Dependencies

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20manager-6A5ACD)
![Make](https://img.shields.io/badge/Make-automation-064F8C)

## Requirements

- Windows PowerShell
- `uv` installed
- `make` installed

## Setup

From the project root:

```powershell
make setup
```

This command creates `.venv`, updates `uv.lock`, and syncs dependencies (including dev tools).

## Common Commands

```powershell
make help         # list available targets
make setup        # create venv, lock, and sync (including dev)
make sync         # sync dependencies
make sync-dev     # sync dependencies + dev
make lock         # update uv.lock
make add PKG=...  # add dependency
make notebook     # open Jupyter Lab via uv

make lint         # run ruff check
make format       # run ruff format
make lint-format  # run lint then format

make precommit-install
make precommit-run

make notebooks          # run notebooks sequentially (output/runs/<run_id>)
make notebooks-inplace  # run notebooks in-place
make notebooks-continue # continue even if one notebook fails

make fetch-news OUTPUT=data/newsdata_news.csv LANGUAGE=pt MAX_RECORDS=200
make clean
```

## Linting and Formatting

The project uses Ruff for linting and formatting.

- Local lint: `make lint`
- Local format: `make format`
- Combined run: `make lint-format`

Ruff is also configured in pre-commit:

- `ruff` (with `--fix`)
- `ruff-format`
- `nbstripout` for notebooks

If pre-commit and local Ruff ever disagree, update the pre-commit Ruff revision and run:

```powershell
uv run pre-commit clean
uv run pre-commit run --all-files
```

## Environment Variables (`.env`)

Create `.env` from the template:

```powershell
Copy-Item .env.example .env
```

Example:

```dotenv
# LLM providers
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
DEEPSEEK_API_KEY=your_deepseek_key

# Optional OpenRouter metadata
OPENROUTER_HTTP_REFERER=https://your-project-or-website.example
OPENROUTER_X_TITLE=llm-misinformation-simulation

# Local OpenAI-compatible endpoint (optional)
LOCAL_OPENAI_API_KEY=ollama
LOCAL_OPENAI_BASE_URL=http://127.0.0.1:11434/v1

# NewsData.io
NEWSDATA_API_KEY=your_newsdata_key
```

Variable reference:

- `GEMINI_API_KEY`: Required for `Provider.GEMINI`
- `OPENROUTER_API_KEY`: Required for `Provider.OPENROUTER`
- `DEEPSEEK_API_KEY`: Required for `Provider.DEEPSEEK`
- `OPENROUTER_HTTP_REFERER`: Optional OpenRouter header
- `OPENROUTER_X_TITLE`: Optional OpenRouter header (`misinformation-llm-simulation` default)
- `LOCAL_OPENAI_API_KEY`: Optional for `Provider.LOCAL` (`ollama` default)
- `LOCAL_OPENAI_BASE_URL`: Optional for `Provider.LOCAL` (`http://127.0.0.1:11434/v1` default)
- `NEWSDATA_API_KEY`: Required for `make fetch-news`

Note: You can still pass `api_key=` and `base_url=` directly in `rewrite_news_with_personality`.

## Main Notebook Workflow

Main simulation notebook:

- [src/llm_simulation_workbench.ipynb](src/llm_simulation_workbench.ipynb)

It:

- loads and organizes datasets,
- rewrites content with multiple providers,
- exports rewritten datasets for audit.

Providers are defined in:

- [src/enums/providers.py](src/enums/providers.py)
- [src/enums/models.py](src/enums/models.py)

Open Jupyter:

```powershell
make notebook
```

## Sequential Notebook Execution

Script:

- [src/run_notebooks_sequentially.py](src/run_notebooks_sequentially.py)

Default run:

```powershell
make notebooks
```

Useful options:

```powershell
make notebooks-continue
make notebooks-inplace
make notebooks NOTEBOOKS="src/llm_simulation_workbench.ipynb src/bert_fake_real_workbench.ipynb"
```

Execution report paths:

- Sequential runs: `output/runs/<run_id>/execution_report.md`
- Manual notebook execution (outside run orchestration): `output/execution_report.md`

## Fetch News with NewsData.io

Script:

- [src/fetch_newsdata_to_csv.py](src/fetch_newsdata_to_csv.py)

Example:

```powershell
make fetch-news OUTPUT=data/newsdata_news.csv LANGUAGE=pt MAX_RECORDS=200
```

With filters:

```powershell
make fetch-news QUERY=politics COUNTRY=us CATEGORY=politics LANGUAGE=en MAX_RECORDS=300 OUTPUT=data/newsdata_politics_us.csv
```

Main arguments:

- `OUTPUT`: output CSV path (`data/newsdata_news.csv` default)
- `QUERY`: search text (`q` API parameter)
- `LANGUAGE`: language(s), e.g. `en` or `pt,en`
- `COUNTRY`: country code(s), e.g. `br` or `br,us`
- `CATEGORY`: category(ies), e.g. `politics,technology`
- `MAX_RECORDS`: max number of records

## Audits

### Consistency Audit (NLI)

Notebook:

- [src/bert_fake_real_workbench.ipynb](src/bert_fake_real_workbench.ipynb)

It computes entailment/contradiction between original and rewritten text and assigns:

- `consistent_with_original`
- `potentially_false_after_rewrite`

### Pretrained Fake News Detector Audit

Notebook:

- [src/pretrained_fake_news_detector_workbench.ipynb](src/pretrained_fake_news_detector_workbench.ipynb)

It applies a pretrained detector to rewritten text and exports per-dataset and consolidated predictions.

## Output Structure

The `output/` directory has two patterns:

1. Latest/manual outputs (shared folders)
2. Run-scoped outputs under `output/runs/<run_id>/`

Typical structure:

```text
output/
	execution_report.md                        # manual notebook execution report (when not using run_id)
	rewritten/
		*.csv                                    # latest rewritten datasets
	audit/
		LocalAudit/
			*_consistency_audit.csv
			all_datasets_consistency_audit.csv
			audit_summary.csv
		PreTrainedAudit/
			*_pretrained_fake_news_predictions.csv
			all_datasets_pretrained_fake_news_predictions.csv
			pretrained_fake_news_summary.csv
	runs/
		<run_id>/
			execution_report.md
			rewritten/
				*.csv
			audit/
				LocalAudit/
					*.csv
				PreTrainedAudit/
					*.csv
			executed_notebooks/
				*.ipynb
```

### Example: single run

If `run_id = 20260322_185309`, the main outputs are usually:

- `output/runs/20260322_185309/execution_report.md`
- `output/runs/20260322_185309/rewritten/local_llama_rewritten_df.csv`
- `output/runs/20260322_185309/audit/LocalAudit/all_datasets_consistency_audit.csv`
- `output/runs/20260322_185309/audit/PreTrainedAudit/all_datasets_pretrained_fake_news_predictions.csv`
- `output/runs/20260322_185309/executed_notebooks/llm_simulation_workbench.ipynb`

## CSV Column Guide

This section highlights the most important CSV columns in the pipeline, with emphasis on columns created and analyzed by notebooks.

### 1) Input dataset columns

| Column | Where it appears | Why it matters |
| --- | --- | --- |
| `full_description` | raw dataset / fetched news | Preferred long-text source for rewriting when available |
| `content` | raw dataset / fetched news | Secondary long-text source |
| `description` | raw dataset / fetched news | Common source text for rewriting and audits |
| `title` | raw dataset / fetched news | Fallback source text and prompt context |
| `language` | raw dataset / fetched news | Helps choose output language in rewriting |
| `country` | raw dataset / fetched news | Additional signal for output language selection |
| `category` | raw dataset / fetched news | Dataset profiling and report summaries |
| `keywords` | raw dataset / fetched news | Dataset profiling and report summaries |
| `source_name` | raw dataset / fetched news | Dataset profiling and source diversity summaries |
| `article_id` | fetched news CSV | Record identity and metadata-row marker (`__query_metadata__`) |

### 2) Columns created during rewriting (`llm_simulation_workbench.ipynb`)

| Column | Created by | Meaning |
| --- | --- | --- |
| `rewritten_news` | `rewrite_news_with_personality` | Final rewritten text |
| `rewrite_status` | `rewrite_news_with_personality` | Rewrite status (`success`, `error`, `skipped`, etc.) |
| `rewrite_error` | `rewrite_news_with_personality` | Error details when rewrite fails |
| `source_text_column` | `rewrite_news_with_personality` | Which source text column was actually used |
| `target_language` | `rewrite_news_with_personality` | Output language code selected for rewriting |
| `target_language_source` | `rewrite_news_with_personality` | Why language was selected (`row.language`, `row.country`, `heuristic`, `default`) |

### 3) Columns created in consistency audit (`bert_fake_real_workbench.ipynb`)

| Column | Created by | Meaning |
| --- | --- | --- |
| `entailment` | NLI scoring | Probability that rewrite is supported by original |
| `contradiction` | NLI scoring | Probability that rewrite contradicts original |
| `neutral` | NLI scoring | Neutral probability |
| `consistency_flag` | `consistency_flag(...)` | Final label (`consistent_with_original` or `potentially_false_after_rewrite`) |
| `row_index` | audit loop | Row reference in source dataframe |
| `row_id` | optional from input | Preserved custom identifier (if configured) |
| `dataset_name` | notebook | Dataset identifier used in grouping and exports |
| `source_file` | notebook | Original CSV filename for traceability |

### 4) Columns created in pretrained detector audit (`pretrained_fake_news_detector_workbench.ipynb`)

| Column | Created by | Meaning |
| --- | --- | --- |
| `prediction_id` | detector inference | Predicted class id |
| `prediction_label` | detector inference | Predicted class label |
| `prediction_confidence` | detector inference | Confidence for predicted class |
| `row_index` | audit loop | Row reference in source dataframe |
| `dataset_name` | notebook | Dataset identifier used in grouping and exports |
| `source_file` | notebook | Original CSV filename for traceability |

### 5) Summary CSV columns

| File | Key columns |
| --- | --- |
| `audit_summary.csv` | `dataset_name`, `source_file`, `rows`, `suspects`, `suspect_rate` |
| `pretrained_fake_news_summary.csv` | `dataset_name`, `source_file`, `rows`, `fake_rate` |

### 6) Metadata row in fetched CSVs

Fetched files may contain one special row where:

- `article_id = __query_metadata__`
- `title = QUERY_METADATA`
- `description` stores a JSON payload with request history and aggregated dataset summary

This row is useful for provenance and query auditing, and should not be treated as a normal news record.

## Pre-commit Notes

Install and run:

```powershell
make precommit-install
make precommit-run
```

If hooks modify files, stage again and re-run before commit.

## Interpretation Notes

- High `contradiction` with low `entailment` increases factual distortion risk after rewriting.
- Rows flagged as `potentially_false_after_rewrite` should be manually reviewed.
