# LLM Misinformation Simulation Workbench

An LLM-based misinformation simulation framework with a rewrite, export, and factual-audit pipeline.

## Required Dependencies

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

This command creates `.venv`, updates `uv.lock`, and syncs dependencies (including the dev group).

## Makefile Commands

```powershell
make help       # list available targets
make setup      # create venv, lock, and sync
make sync       # sync dependencies
make sync-dev   # sync dependencies + dev
make lock       # update uv.lock
make add PKG=package-name
make notebook   # open Jupyter Lab via uv
make precommit-install
make precommit-run
make notebooks          # run notebooks sequentially (output/runs/<run_id>)
make notebooks-inplace  # run notebooks in-place
make notebooks-continue # continue even if one notebook fails
make fetch-news OUTPUT=data/newsdata_news.csv LANGUAGE=pt MAX_RECORDS=200
make clean      # remove .venv
```

## Environment Variables (.env)

Create a `.env` file in the project root and define the variables you need for your chosen providers and data sources.

Use [.env.example](.env.example) as the canonical template:

```powershell
Copy-Item .env.example .env
```

### Full example

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

# NewsData.io (used by make fetch-news)
NEWSDATA_API_KEY=your_newsdata_key
```

### Variable reference

- `GEMINI_API_KEY`: Required when using `Provider.GEMINI`.
- `OPENROUTER_API_KEY`: Required when using `Provider.OPENROUTER`.
- `DEEPSEEK_API_KEY`: Required when using `Provider.DEEPSEEK`.
- `OPENROUTER_HTTP_REFERER`: Optional header for OpenRouter requests.
- `OPENROUTER_X_TITLE`: Optional OpenRouter title header. Default: `misinformation-llm-simulation`.
- `LOCAL_OPENAI_API_KEY`: Optional key for `Provider.LOCAL`. Default: `ollama`.
- `LOCAL_OPENAI_BASE_URL`: Optional base URL for `Provider.LOCAL`. Default: `http://127.0.0.1:11434/v1`.
- `NEWSDATA_API_KEY`: Required by `make fetch-news` (used by `src/fetch_newsdata_to_csv.py`).

Note: In `rewrite_news_with_personality`, you can also pass `api_key=` and `base_url=` directly, which overrides `.env` values for that call.

## Main Simulation Notebook

- [src/llm_simulation_workbench.ipynb](src/llm_simulation_workbench.ipynb)

This notebook does the following:

- load and organize datasets,
- rewrite content with multiple LLM providers,
- export results to [output/runs/<run_id>/](output/runs).

Providers are configured through the enum in [src/enums/providers.py](src/enums/providers.py) and used in the notebook as `Provider.GEMINI`, `Provider.OPENROUTER`, and `Provider.LOCAL`.

To open Jupyter:

```powershell
make notebook
```

## Run Notebooks Sequentially

Use [src/run_notebooks_sequentially.py](src/run_notebooks_sequentially.py) to execute notebooks in order.

Each run now gets a unique `run_id` and is stored under `output/runs/<run_id>/`.

Default flow (LLM simulation + BERT audit):

```powershell
make notebooks
```

Useful options:

```powershell
# Continue even if one notebook fails
make notebooks-continue

# Run in-place
make notebooks-inplace

# Choose custom notebook order
make notebooks NOTEBOOKS="src/llm_simulation_workbench.ipynb src/bert_fake_real_workbench.ipynb"
```

## Fetch News with NewsData.io

The script [src/fetch_newsdata_to_csv.py](src/fetch_newsdata_to_csv.py) fetches news via the NewsData.io API and saves a CSV file to [data/](data/).

1. Configure your key in `.env`:

```powershell
NEWSDATA_API_KEY=your_key_here
```

2. Run via Makefile:

```powershell
make fetch-news OUTPUT=data/newsdata_news.csv LANGUAGE=pt MAX_RECORDS=200
```

Example with additional filters:

```powershell
make fetch-news QUERY=politica COUNTRY=br CATEGORY=politics LANGUAGE=pt MAX_RECORDS=300 OUTPUT=data/newsdata_politics_br.csv
```

Main arguments:

- `OUTPUT`: output CSV path (default: `data/newsdata_news.csv`)
- `QUERY`: search text (`q` API parameter)
- `LANGUAGE`: language(s), e.g. `pt` or `pt,en`
- `COUNTRY`: country code(s), e.g. `br` or `br,us`
- `CATEGORY`: category(ies), e.g. `politics,technology`
- `MAX_RECORDS`: maximum number of records to save

## Factuality Audit After Rewriting (Real-News Scenario)

If your original data is fully real, the recommended flow is checking whether the rewrite contradicts facts from the original text.

For that, use:

- `src/bert_fake_real_workbench.ipynb`

This notebook:

- reads CSVs generated in [output/runs/<run_id>/](output/runs) by the simulation notebook,
- uses an NLI model to compute `entailment` and `contradiction`,
- labels each row as:
	- `consistente_com_original`, or
	- `potencialmente_falsa_apos_reescrita`.

### How to Run

1. Open Jupyter:

```powershell
make notebook
```

2. Open `src/bert_fake_real_workbench.ipynb`.
3. In Cell 5 (configuration), adjust:

- `INPUT_DIR`
- `DATASET_SELECTOR`
- `ORIGINAL_COLUMN`
- `REWRITTEN_COLUMN`
- optionally `ROW_ID_COLUMN`

4. Run all notebook cells in order.

### Output


- CSV files in [results/](results):
	- `all_datasets_consistency_audit.csv`
	- `audit_summary.csv`
	- one file per dataset (`*_consistency_audit.csv`)
- main columns:
	- `entailment`
	- `contradiction`
	- `neutral`
	- `consistency_flag`

## Pre-commit and Notebooks

To install and run hooks:

```powershell
make precommit-install
make precommit-run
```

For notebooks, the hook can modify files and fail on the first attempt. In that case:

```powershell
git add -A
make precommit-run
git add -A
git commit -m "your message"
```

### Interpretation

- High `contradiction` with low `entailment` indicates a higher risk of factual distortion after rewriting.
- Cases with `consistency_flag = potencialmente_falsa_apos_reescrita` should be reviewed manually.
