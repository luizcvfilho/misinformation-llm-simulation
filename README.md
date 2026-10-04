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

make fetch-news OUTPUT=data/raw/newsdata_news.csv LANGUAGE=pt MAX_RECORDS=200
make interaction-graph
make interaction-graph-verbose GRAPH_MAX_ROWS=5
make interaction-graph-ui
make interaction-graph-analysis
make interaction-graph-analysis-ui
make interaction-graph-dashboards
make clean
```

### Brazilian elections corpus exploration

Open the local exploratory notebook with:

```powershell
uv sync --locked
uv run jupyter lab notebooks/elections_2026_eda.ipynb
```

It reads `data/Eleições 2026/`, including the three post Parquets, the nested Meta Ad Library CSV,
and the BRPOL aggregate ZIP. It audits schemas, missingness, duplicate IDs and captions, publication
dates, source groups, candidate metadata, account concentration, observed engagement, lexical
patterns, ad range endpoints, and TSE collection coverage. Organic posts, ads, and overlapping
aggregates are analyzed separately. Caption topic exploration uses a bounded local sample and
does not call an LLM or download models.

The configuration cell controls ad start-date filters and reproducible simulation candidate
sampling. Local outputs in `output/elections_2026/` include CSV summaries, interactive HTML charts,
`simulation_candidates.csv`, and a manifest with input hashes and parameters. Candidates require
manual review for election relevance and sufficient standalone context. The candidate CSV supports
the interaction-graph workflow with `description` as the text column; `category` denotes the source
group. Review the source package README's use conditions before sharing data or executed outputs.
The notebook does not run simulations.

### Interaction-graph STDI analysis

The persisted runs in `output/interaction_graph/app_runs/` can be analyzed without rerunning the
LLM simulation. The analysis command recursively reads `*_steps.jsonl` files and always ignores
directories named `OLD_RUNS` (case-insensitive):

```powershell
make interaction-graph-analysis
```

It writes self-contained interactive Plotly HTML figures, tidy successful step data, summaries by
iteration and chain, and a reproducibility manifest to `output/interaction_graph/analysis/`. When
Kaleido and Chrome or Chromium are available, PNG copies are generated as well; the
`plotly_figure_export.json` file records whether this export succeeded. The figures use
`stdi_vs_original`, because it measures each rewritten version against the same original news
item. The dashboard also lets the user inspect `stdi_incremental` and `stdi_cumulative`.

To interactively filter chains and metrics, inspect the component trajectories, and move the
boxplot iteration selector, run:

```powershell
make interaction-graph-analysis-ui
```

The dashboard uses Plotly, including a visible mode bar for zoom, pan, box and lasso selection,
scale reset, and figure export.

Enter one or more paths in **Run folders (one per line)**. Paths may point to individual execution
folders or parent folders; overlapping paths load each step file only once. Select the executions
to inspect in **Executions**, then switch between them with **Active execution**. Executions are
listed newest first, and the most recent selected execution is the default. Recency uses the
timestamp in standard `simulation_ui_YYYYMMDD_HHMMSS` folder names, falling back to the folder's
modification time for custom names. Every chart,
summary, paired comparison, and case uses only that execution, so repeated news IDs and chain
names in different executions are kept separate.

Chain names are read from persisted filenames, including custom names with underscores, spaces,
or Unicode characters. The original persona-code names remain compatible. In **Chain comparisons**,
choose **Custom pairs**, select **Chain A**, and choose one or more chains B to compare final scores
for shared news items. Custom pairs also work in the qualitative contrast case explorer. Preset
contrasts remain available when their chains are present.

### Running both Streamlit dashboards

The workflow UI and the analysis dashboard use distinct ports by default, so they can run at the
same time. Start both from one terminal with:

```powershell
make interaction-graph-dashboards
```

Then open `http://localhost:8501` for the interaction graph workflow and
`http://localhost:8502` for the analysis dashboard. Press `Ctrl+C` in that terminal to stop both
processes.

To start them from separate terminals instead, use:

```powershell
make interaction-graph-ui
make interaction-graph-analysis-ui
```

Override the ports when either default is already in use:

```powershell
make interaction-graph-dashboards GRAPH_UI_PORT=8511 GRAPH_ANALYSIS_UI_PORT=8512
```

## Structured Topic Drift Index (STDI)

STDI compares a rewritten news item with a reference text using four content components,
an internal-contradiction contribution, and an optional VAD contribution. It measures
textual drift, not factual veracity, belief, exposure, or sharing behavior.

The extraction step builds a structured representation for each text with:

- `main_topic`
- `subtopics`
- `central_entities`
- `central_relations` in `(subject, action, object)` form
- `narrative_frame` (optional)
- `has_internal_contradiction` (boolean)
- `internal_contradiction_score` from 0 to 1

### Content comparison methods

The four content components depend on the comparison method. **Interaction-graph
simulations default to `cluster`**, which uses local embeddings. Calling
`calculate_stdi(...)` without `component_overrides` uses the lexical method;
`calculate_stdi_chain_metrics(...)`, `annotate_stdi_for_rewrites(...)`, and
`annotate_stdi_for_version_chain(...)` also use lexical comparison. The manual
evaluation workflow uses `llm_semantic` scores for these four components.

For `cluster`, [ClusterSTDIComparator](src/misinformation_simulation/topic_drift/cluster_comparison.py)
embeds the extracted labels and relations with
`sentence-transformers/all-MiniLM-L6-v2`. Scalar similarity `S(a, b)` is cosine
similarity clipped to `[0, 1]`; normalized equal non-empty labels have similarity 1,
and a comparison involving an empty label has similarity 0.

- **Theme:** `D_theme = 1 - S(main_topic_reference, main_topic_version)`.
  Normalized equal main topics have drift 0, including two missing main topics.
- **Subtopics and entities:** compute all pairwise similarities, sort them in
  descending order, and greedily select matches without reusing an item on either
  side. Divide the sum of selected similarities by the larger list size, then
  subtract from 1. Unmatched items therefore increase drift. Two empty lists have
  drift 0; one empty list has drift 1.
- **Relations:** use the same greedy matching, with the relation-pair similarity
  defined below. `S_triple` compares the serialized
  `subject: ... | action: ... | object: ...` texts; the other terms compare the
  corresponding subject, action, and object labels.

```text
S_relation = 0.45*S_triple + 0.25*S_subject + 0.15*S_action + 0.15*S_object
D_list = 1 - sum(selected_pair_similarities) / max(reference_count, version_count)
```

The comparator fits shared KMeans clusters over the run's collection, but cluster
identifiers are diagnostic outputs: the scores use direct embedding similarities
and greedy matching, not Jaccard over cluster IDs or a same-cluster test.

For the **lexical** method, labels are lowercased and whitespace is normalized.
Subtopics and entities are compared as sets; relations are sets of normalized
`(subject, action, object)` triples:

```text
D_theme = 0 if main_topic is equal, else 1
D_subtopic = 1 - Jaccard(subtopics_reference, subtopics_version)
D_entities = 1 - Jaccard(entities_reference, entities_version)
D_relations = 1 - Jaccard(relations_reference, relations_version)
```

Two empty sets have drift 0; one empty set has drift 1.
For **`llm_semantic`**, an LLM evaluates both texts
and their structures together, assigning each content component one of
`0`, `0.25`, `0.5`, `0.75`, or `1`, with a rationale.

### Shared aggregation, contradiction, and VAD

All methods use the same aggregation in
[calculate_stdi](src/misinformation_simulation/topic_drift/metrics.py). Embedding or
LLM component scores replace the four lexical content values through
`component_overrides`; their default weights remain equal:

```text
content_drift =
  0.25*D_theme +
  0.25*D_subtopic +
  0.25*D_entities +
  0.25*D_relations
```

`D_contradiction` is the **compared version's** `internal_contradiction_score`,
clipped to `[0, 1]`. Despite the output name `contradiction_drift`, it is not the
difference between the reference and version scores. The boolean
`has_internal_contradiction` and `narrative_frame` are not separate numeric terms.
This contribution assesses contradiction within the version, not disagreement
with the reference or external facts. Consequently, STDI is not necessarily
symmetric, and an identical text with a nonzero contradiction score can have
nonzero STDI.

For VAD, the default score range is 4, corresponding to scores on the project's
1–5 scale. `score_range` can override this divisor:

```text
D_valence = min(abs(valence_version - valence_reference) / score_range, 1)
D_arousal = min(abs(arousal_version - arousal_reference) / score_range, 1)
D_dominance = min(abs(dominance_version - dominance_reference) / score_range, 1)
D_vad = (D_valence + D_arousal + D_dominance) / 3

contradiction_increment = (1 - content_drift) * 0.20 * D_contradiction
drift_with_contradiction = content_drift + contradiction_increment
STDI = drift_with_contradiction + (1 - drift_with_contradiction) * 0.20 * D_vad
```

Contradiction and VAD each add at most 20% of the remaining distance, sequentially;
they are not two additional terms in a six-component weighted average. For example,
`content_drift = 0.4`, `D_contradiction = 0.5`, and `D_vad = 0.25` give
`drift_with_contradiction = 0.46` and `STDI = 0.487`. Pair-level STDI stays in
`[0, 1]`; complete content drift gives STDI 1 regardless of contradiction or VAD.
Returned metrics are rounded to six decimal places; VAD dimensions are also
rounded before their mean is computed.

At the low-level metric API, a missing VAD object makes all VAD drifts 0; a missing
dimension contributes 0 to the mean, which still divides by 3. This is a fallback
in the implementation, not evidence of emotional neutrality. Graph simulations
require all three VAD scores for each evaluated text.

Available helpers in [src/misinformation_simulation/topic_drift](src/misinformation_simulation/topic_drift):

- `extract_topic_structure(...)`
- `ClusterSTDIComparator`
- `calculate_stdi(...)`
- `calculate_vad_drift(...)`
- `calculate_stdi_chain_metrics(...)`
- `build_manual_stdi_evaluation_dataset(...)`
- `generate_metric_rewrites(...)`
- `score_manual_stdi_evaluation_pairs(...)`
- `summarize_manual_stdi_evaluation(...)`
- `fit_manual_stdi_regression(...)`
- `compare_stdi_components_semantically(...)`
- `annotate_stdi_for_rewrites(...)`
- `annotate_stdi_for_version_chain(...)`

Example with the lexical annotation flow (this helper does not use the graph's
default embedding comparator):

```python
from misinformation_simulation.topic_drift import annotate_stdi_for_rewrites

rewritten_with_stdi = annotate_stdi_for_rewrites(
    df=rewritten_df,
    rewritten_column="rewritten_news",
    provider="chatgpt",
    model="gpt-6-luna",
    vad_model_bundle=vad_model,
)
```

Sequential rewrite chains report:

- `stdi_vs_original`: each version compared with the original article
- `stdi_incremental`: each version compared with the previous successfully evaluated version
- `stdi_cumulative`: running sum of incremental STDI values along the chain

The cumulative sum is not normalized and can exceed 1; it is not the distance
from the original. Failed or skipped versions do not add to this sum.

Generated columns include the extracted structures for the original and each version,
their VAD scores, and the per-version STDI metrics.

### Manual STDI evaluation and calibration

The calibration workflow now uses 50 real news items sampled reproducibly from
`data/raw/newsdata_news.csv`. It creates one rewritten version of each news item and assigns
one of six controlled prompts: main theme, subtopic, central entity, central relation,
internal contradiction, or emotional framing (VAD). The 50 pairs are distributed as
evenly as possible across the six target metrics.

Generate the review set, rewrites, and calculated STDI values with:

```powershell
make prepare-stdi-manual-evaluation STDI_MANUAL_GENERATE=1 STDI_MANUAL_SCORE=1
```

The default model is `gpt-6-luna`, the provider is `chatgpt`, and the limit is 450
requests per minute. This leaves headroom below the Tier 1 500 RPM limit for retries and
other API traffic. Set `CHATGPT_API_KEY` or `OPENAI_API_KEY` in `.env` first.
The source sampler excludes the promotional text `ONLY AVAILABLE IN PAID PLANS`, bodies with
fewer than 50 words, terminal ellipses, and common continuation markers such as `Read more`.
Rewrite prompts instruct the model to keep the body between 85% and 115% of the original word
count; generated text is not automatically retried or rejected based on its final length.

For `vad_drift`, the rewrite prompt explicitly requests lower valence and higher arousal.
The generated text is scored together with all other pairs after rewriting, without an
automatic acceptance threshold or additional rewrite attempts.

The command writes its files to `output/stdi_manual_evaluation/`:

- `manual_stdi_pairs.csv`: sampled real news items and the six assigned prompts.
- `generated_stdi_pairs.csv`: generated rewrites.
- `scored_stdi_pairs.csv`: semantic STDI components, lexical reference values, rationales, and
  empty manual-review columns.
- `stdi_evaluation_summary.csv`: calculated results grouped by target metric.
- `manual_review_guide.md`: annotation rubric.

Review `scored_stdi_pairs.csv`, filling `manual_target_metric_score`,
`manual_expected_stdi`, `manual_review_status`, and `manual_notes`. Then fit the
weights from the manually assigned overall STDI values:

```powershell
make calibrate-stdi
```

This creates `calibrated_stdi_weights.csv` and `calibration_metrics.json` in the same
directory. Regression features are the calculated STDI components, while the target is
the manually reviewed `manual_expected_stdi` value.
Fitting exports regression coefficients and normalized weights for analysis; it
does not automatically replace the constants used by `calculate_stdi(...)`.

### CSV Explorer

Open a generic local CSV explorer with:

```powershell
make csv-explorer
```

Load any CSV file, select its delimiter, filter values, choose visible columns, inspect a
single row vertically, and download a filtered copy. The original CSV is not modified.

Use `Compare CSVs` to load two files, choose the join key for each source, select the
columns to compare, and display matching fields side by side. Result columns are prefixed
and color-coded by source, while filters, charts, row details, and downloads remain
available for the joined data.

Each scored pair makes one additional semantic-comparison request. This evaluator compares
the two news items together and scores theme, subtopic, entity, and relation drift using
the fixed levels `0`, `0.25`, `0.5`, `0.75`, and `1`. Its explanations are written to the
`semantic_*_rationale` columns. Contradiction and VAD continue to use their existing metrics.

### Reusable LLM and clustering comparisons

The project can compare the same LLM-extracted structures using two methods:

- `llm_semantic`: an LLM judges theme, subtopic, entity, and relation drift.
- `cluster`: all extracted labels and triples are embedded with
  `sentence-transformers/all-MiniLM-L6-v2`, clustered globally for the run, and compared
  using the direct similarities and greedy matching described above. The extraction
  step remains shared with `llm_semantic`.

Both methods read the same pair schema (`original_text`, `modified_text`, and optional
`original_*` / `modified_*` extracted fields) and produce `comparison_results.csv` with the
same base drift columns. A result folder can be re-used as input for the other method.
This reusable comparison workflow calls `calculate_stdi(...)` without VAD objects,
so its VAD drift columns are 0 and its STDI includes content and contradiction
only. The manual evaluation and graph workflows supply VAD separately.

```powershell
# First method. Missing structures are extracted once by the configured LLM.
make topic-drift-comparison TOPIC_DRIFT_COMPARISON_ARGS="--input output/stdi_manual_evaluation/scored_stdi_pairs.csv --output-dir output/topic_drift/llm --method llm_semantic"

# Reuse the structures and texts from the first output, with no new extraction calls.
make topic-drift-comparison TOPIC_DRIFT_COMPARISON_ARGS="--input-dir output/topic_drift/llm --output-dir output/topic_drift/cluster --method cluster --compare-with output/topic_drift/llm"
```

The second command writes `method_comparison.csv`, joined by `pair_id`, and cluster assignments
under `cluster_artifacts/`. Cluster identifiers are diagnostic and meaningful only
inside the run that fitted them; fit one shared comparator over the complete
collection of original and rewritten structures.

The cluster theme drift is **1 minus** the direct embedding similarity between
`main_topic` labels. The current comparator is recorded as `cluster_v2` in graph
step metadata, graph summaries, and comparison manifests. It no longer extracts
`topic_domain` or forces maximum theme drift based on different domain labels.
The domain-coverage audit command and domain-based dashboard grouping have been
retired; news grouping uses the original dataset's `category` field.

Historical files retain their original scores. Reusing saved structures recomputes
cluster scores with the current comparator and ignores legacy domain fields;
it does not require new extraction requests. The analysis loader marks runs without
a comparison version as `legacy`. Treat historical scores and `cluster_v2` scores
as different measurement versions when comparing experiments.

To regenerate the extracted structures as well, use:

```powershell
make topic-drift-comparison TOPIC_DRIFT_COMPARISON_ARGS="--input-dir output/topic_drift/previous_run --output-dir output/topic_drift/refreshed_cluster --method cluster --refresh-structures"
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

### Rewriting false news as truthful news

`rewrite_false_news_as_true` is the project's central implementation for rewriting false or
unsupported news into neutral, truthful versions. The topic-matched VAD analysis uses this same
function, and it can also be called directly with any compatible DataFrame.

It expects an `original_article_text` column and returns a copy with the generated article in
`rewritten_article_text`, along with the prompt, model, provider, status, and error columns.

```python
import pandas as pd
from misinformation_simulation.llm import rewrite_false_news_as_true

false_news = pd.read_csv("false_news.csv")
rewritten = rewrite_false_news_as_true(
    false_news,
    text_column="original_article_text",
    topic_column="subject",
    title_column="title",
    provider="chatgpt",
    model="gpt-6-luna",
    checkpoint_path="false_news_rewritten_as_true.csv",
)
```

With `checkpoint_path`, the CSV is saved initially, every ten attempted rewrites, and at the end
of the call. Writes are atomic and retried to reduce failures caused by transient file locks. A
later call can then resume without reprocessing rows whose `rewrite_status` is `success`, including
after an API quota error or an interrupted execution.

The function asks the model to correct or remove unsupported claims, but it does not perform
external fact-checking. Validate factual claims independently before treating an output as true.

### STDI logistic-regression analysis

The resumable workflow below rewrites a false-news CSV, reuses successful rewrites and STDI
annotations already present in the selected output directory, and creates an interpretable
logistic-regression analysis. Its features are numeric representations of the STDI structures:
counts of subtopics, entities, and relations, internal contradiction, VAD dimensions, and text
length. A separate pair-level CSV preserves STDI and delta features for each false-to-rewrite pair.

```powershell
make stdi-logistic-regression STDI_REGRESSION_INPUT=data/FakeVsTrueVAD/Fake.csv STDI_REGRESSION_TEXT_COLUMN=text STDI_REGRESSION_OUTPUT_DIR=output/stdi_logistic_regression
```

For each row in `Fake.csv`, the workflow uses the original false article and its own truthified
rewrite as a controlled pair. These are the reference classes used to study which STDI-derived
features distinguish the versions; the resulting model is not a factual verifier. The optional
`STDI_REGRESSION_TRUE_REFERENCE_INPUT` adds an external reference group, but is not needed for the
paired analysis. The output directory contains STDI audits, the regression dataset, pair metrics,
feature importance, model metrics, a manifest, and a short report.

Set `STDI_REGRESSION_SKIP_REWRITE=1` to reuse the truthified CSV in later runs. If that CSV
does not yet exist in the output directory, the workflow creates it automatically.

The workflow also fits a lexical comparison with TF-IDF unigrams and bigrams. It writes
`stdi_tfidf_model_comparison.csv`, comparing `stdi_only`, `tfidf_only`, and
`stdi_plus_tfidf` on the same group-safe holdout split, plus `stdi_tfidf_top_ngrams.csv` with
the strongest lexical coefficients. This keeps lexical performance separate from the structural
STDI importance report. The defaults are 10,000 terms, `min_df=5`, and an `(1, 2)` n-gram range;
adjust the first two with `STDI_REGRESSION_TFIDF_MAX_FEATURES` and
`STDI_REGRESSION_TFIDF_MIN_DF`.

## Main Notebook Workflow

Main simulation notebook:

- [notebooks/llm_simulation_workbench.ipynb](notebooks/llm_simulation_workbench.ipynb)

It:

- loads and organizes datasets,
- rewrites content with multiple providers,
- exports rewritten datasets for audit.

Providers are defined in:

- [src/misinformation_simulation/enums/providers.py](src/misinformation_simulation/enums/providers.py)
- [src/misinformation_simulation/enums/models.py](src/misinformation_simulation/enums/models.py)

Open Jupyter:

```powershell
make notebook
```

## Sequential Notebook Execution

Script:

- [scripts/run_notebooks.py](scripts/run_notebooks.py)

Default run:

```powershell
make notebooks
```

Useful options:

```powershell
make notebooks-continue
make notebooks-inplace
make notebooks NOTEBOOKS="notebooks/llm_simulation_workbench.ipynb notebooks/bert_fake_real_workbench.ipynb"
```

Execution report paths:

- Sequential runs: `output/runs/<run_id>/execution_report.md`
- Manual notebook execution (outside run orchestration): `output/execution_report.md`

## Fetch News with NewsData.io

Script:

- [scripts/fetch_newsdata.py](scripts/fetch_newsdata.py)

Example:

```powershell
make fetch-news OUTPUT=data/raw/newsdata_news.csv LANGUAGE=pt MAX_RECORDS=200
```

With filters:

```powershell
make fetch-news QUERY=politics COUNTRY=us CATEGORY=politics LANGUAGE=en MAX_RECORDS=300 OUTPUT=data/raw/newsdata_politics_us.csv
```

Main arguments:

- `OUTPUT`: output CSV path (`data/raw/newsdata_news.csv` default)
- `QUERY`: search text (`q` API parameter)
- `LANGUAGE`: language(s), e.g. `en` or `pt,en`
- `COUNTRY`: country code(s), e.g. `br` or `br,us`
- `CATEGORY`: category(ies), e.g. `politics,technology`
- `MAX_RECORDS`: max number of records

## Interaction Graph

Script:

- [scripts/run_interaction_graph.py](scripts/run_interaction_graph.py)

This workflow runs a chained LLM interaction graph over a news dataset using:

- an input file such as `data/graphs/graph_news.csv`
- a graph definition such as `data/graphs/graph_config.json`

Default Make targets:

```powershell
make interaction-graph
make interaction-graph-verbose
make interaction-graph-ui
make interaction-graph-dashboards
```

Useful overrides:

```powershell
make interaction-graph GRAPH_INPUT=data/graphs/graph_news.csv GRAPH_CONFIG=data/graphs/graph_config.json GRAPH_MAX_ROWS=5
make interaction-graph-verbose GRAPH_TEXT_COLUMN=description GRAPH_OUTPUT_PREFIX=politics_graph
```

Main variables:

- `GRAPH_INPUT`: input CSV/JSON/JSONL file (`data/graphs/graph_news.csv` default)
- `GRAPH_CONFIG`: graph JSON config (`data/graphs/graph_config.json` default)
- `GRAPH_TEXT_COLUMN`: text column used by the simulation (`description` default)
- `GRAPH_TITLE_COLUMN`: title column (`title` default)
- `GRAPH_NEWS_ID_COLUMN`: optional custom id column
- `GRAPH_MAX_ROWS`: optional row limit
- `GRAPH_SLEEP_SECONDS`: delay between requests (`0` default)
- `GRAPH_MAX_REQUESTS_PER_MINUTE`: optional rate limit
- `GRAPH_RETRY_ATTEMPTS`: retry attempts (`5` default)
- `GRAPH_ALLOW_TITLE_FALLBACK`: set to any non-empty value to add `--allow-title-fallback`
- `GRAPH_REWRITE_MODE`: `faithful` (default) or `interpretive` (experimental)
- `GRAPH_TOPIC_DRIFT_MODEL`: optional topic drift model override (the application default is
  `gpt-6-luna`)
- `GRAPH_TOPIC_DRIFT_PROVIDER`: optional topic drift provider override (the application default
  is `chatgpt`)
- `GRAPH_OUTPUT_DIR`: output directory (`output/interaction_graph` default)
- `GRAPH_OUTPUT_PREFIX`: output file prefix (`simulation` default)
- `GRAPH_UI_PORT`: Streamlit port for the interaction graph workflow (`8501` default)
- `GRAPH_ANALYSIS_UI_PORT`: Streamlit port for the analysis dashboard (`8502` default)

The script prints a JSON summary and, when available, the generated `summary_path` and `steps_path`.

### Transmission modes

All application prompt definitions live in `src/misinformation_simulation/config/prompts.py`:
the faithful and interpretive modes, personality presets, topic extraction and classification,
controlled rewrites, semantic comparison, and false-to-true rewriting. Other modules import these
definitions; graph JSON files can still supply custom personality text for a particular execution.

The graph keeps **Faithful rewrite (preserve facts)** as its default and control condition.
Its original system instruction, prompt, title context, and generation settings are preserved.
This condition explicitly instructs the model to preserve facts while changing framing and tone.

Select **Interpretive relay (experimental)** under **Execution settings → Transmission mode**
to simulate interpreting a received message and passing it onward. Every node receives only the
preceding message, without a separate copy of the original title. The prompt allows selective
omission, changes in the central issue, and unsupported interpretations of motives, responsibility,
and causality when consistent with the assigned personality. A conspiratorial persona may infer
hidden coordination or invert an official explanation; an evidence-focused skeptic is instead
asked to question weak claims and preserve uncertainty. Existing personality instructions still
apply, including any explicit fact-preservation constraints in custom personas.

The shared interpretive template contains only the transmission rules.
`INTERPRETIVE_PERSONALITY_EXTENSIONS` in `config/prompts.py` supplies the extra behavior for
`ConspiracyDenialist` and `InvestigativeSkeptic`, attaching only the matching extension in the
interpretive mode. Full preset texts and their legacy opening-sentence forms are recognized;
other personality text is used as supplied. The faithful mode keeps the original personality
text. Each step saves `metadata_rewrite_effective_personality` for prompt reconstruction.

The interpretive prompt requires retained names, numbers, dates, and quotations to remain accurate
and prohibits fabricated concrete evidence or specific new events. These are model instructions,
not an automatic factual validation step. The mode applies to all nodes and queued graphs in the
execution; graph JSON files continue to describe the topology and personalities.

For a command-line run:

```powershell
uv run python scripts/run_interaction_graph.py --input data/graphs/graph_politics_news.csv --graph-config data/graphs/graph_config.json --news-id-column article_id --max-rows 5 --rewrite-mode interpretive --output-dir output/interaction_graph/interpretive_pilot --output-prefix interpretive_pilot --verbose
```

The Make workflow also accepts `GRAPH_REWRITE_MODE=interpretive`. Start with the same small news
sample in both modes, using the same models, personas, chain lengths, and evaluation settings.
Compare homogeneous chains with reversed mixed chains, and repeat runs to inspect variation rather
than selecting only extreme outputs. Keep command-line analysis runs for each mode in separate
directories; the Plotly analysis app has a **Transmission mode** filter to select one condition.
Older files without this metadata are labelled **Legacy (mode not recorded)**.

Both modes request temperature `0.8` and use the same STDI/VAD evaluation. The existing OpenAI
request builder omits temperature for `gpt-5` and `gpt-6` model names, leaving the provider default;
the saved `rewrite_temperature_requested` is the configured value, not proof of an applied value.
Topic extraction still receives the original title; only generation title context changes in the
interpretive condition. Summaries
save the mode, prompt version, system instruction, template, temperature, and title-context policy.
Each step records the mode, version, and a hash of its formatted prompt, along with its actual input
and output. Those records make the generation condition identifiable without changing saved older
runs or the standalone article rewriting workflow.

Higher STDI indicates informational drift under the configured metric. It does not by itself
establish external factual falsity, simulated belief or sharing probability, or human realism.
Inspect retained claims, omitted context, invented causal interpretations, and internal
contradictions separately. This mode is an experimental hypothesis about transmission behavior;
its effect on drift and its correspondence to human transmission still require empirical validation.
The curated political dataset mostly supplies titles and descriptions rather than complete article
bodies, which also limits the amount of context available for interpretation.

### Interaction Graph UI

The project also includes a Streamlit interface for the interaction graph workflow:

```powershell
make interaction-graph-ui
```

The UI lets you:

- load a dataset from the project or upload a CSV/JSON/JSONL file
- import an existing graph JSON or define the graph directly in the browser
- add, remove, and reorder nodes in the chain
- add snapshots of multiple graphs to a queue, reorder them, and run them sequentially without further input
- select predefined personalities or write custom personality prompts
- inspect each graph's status, summary, per-node metrics, and per-news outputs
- compare success rates and drift metrics by NewsData `category` across queued graphs and download the comparison as CSV
- compare extracted main topics, subtopics, entities, and relations for the original and rewritten
  text at each node, alongside the drift score for each category

To queue graphs, enter a name and choose **Add current graph**, or expand **Add graphs from a folder** and select a folder of graph JSON configs. Folder import adds valid files in filename order and reports errors for invalid files. You can then edit the graph or
import another JSON config and add it too. **Run graph queue** processes each saved graph against
the selected dataset using the execution settings shown in the UI. A queue writes one batch
folder under the configured output directory, with a dedicated subfolder for every graph. The
batch folder uses the date/time prefix; each graph folder and its files also include the queue
position and graph name (for example,
`app_runs/simulation_ui_20260917_123456/simulation_ui_20260917_123456_01_investigative_skeptic/`).
A single graph remains directly inside its own folder under `app_runs` and uses the editable
**Current graph name**. Repeating a run or batch with the same name and timestamp adds a numeric
suffix so previous results are preserved. If one graph fails, the queue records the error and
continues with the next graph. The Results tab and analysis app search `app_runs` recursively,
so both layouts are loaded from the same default path. When the input has a `category` column,
every step record saves its original
category value as `metadata_category`; the News category comparison in Results separates
semicolon-delimited labels such as `politics; top`. An article contributes to each of its
labels, so category counts overlap. Saved runs can be imported for the same comparison;
older step files without `metadata_category` need their original category data restored or a new run.

Each graph in the queue is a single connected chain of nodes, as required by the backend.

Each graph step computes VAD (valence, arousal, and dominance) for the source article and
the rewritten text using the project's default `RobroKools/vad-bert` model. The STDI
uses the local `sentence-transformers/all-MiniLM-L6-v2` embedding comparator for theme,
subtopic, entity, and relation drift. It fits a shared comparator across the successful
steps in each graph run, then calculates both original and incremental scores using
direct similarities and greedy matching (see the STDI section above). The summary records
`stdi_comparison_method` and `stdi_embedding_model`. The CLI accepts
`--stdi-comparison-method lexical` to reproduce the earlier exact-label calculation.
STDI includes both VAD drift and the existing internal-contradiction contribution. Step records
include these components against the original article and the previous version, together
with the raw VAD scores. The VAD model is loaded locally and may need to be downloaded on
its first use. A graph run requires all three VAD scores for each evaluated text.
Graph step records also include `stdi_cumulative`, the running sum of successful incremental
STDI values for each news item. Failed steps have no cumulative score; a later successful step
continues from the last successful version. This sum is not normalized and can exceed 1.
The UI runs simulations in the background and shows a **Cancel simulation** button while a run
is active. Cancellation takes effect after the current model call or local scoring operation;
completed steps are saved with `cancelled: true` in the summary, and queued graphs that have
not started are skipped.

## Audits

### Consistency Audit (NLI)

Notebook:

- [notebooks/bert_fake_real_workbench.ipynb](notebooks/bert_fake_real_workbench.ipynb)

It computes entailment/contradiction between original and rewritten text and assigns:

- `consistent_with_original`
- `potentially_false_after_rewrite`

### Pretrained Fake News Detector Audit

Notebook:

- [notebooks/pretrained_fake_news_detector_workbench.ipynb](notebooks/pretrained_fake_news_detector_workbench.ipynb)

It applies a pretrained detector to rewritten text and exports per-dataset and consolidated predictions.

### Structured Topic Drift Audit

Notebook:

- [notebooks/topic_drift_audit_workbench.ipynb](notebooks/topic_drift_audit_workbench.ipynb)

It applies STDI to original vs. rewritten news pairs, exports per-dataset and consolidated outputs, and summarizes topic drift using:

- `rewritten_news_stdi_vs_original`
- `rewritten_news_theme_drift_vs_original`
- `rewritten_news_subtopic_drift_vs_original`
- `rewritten_news_entity_drift_vs_original`
- `rewritten_news_relation_drift_vs_original`
- `high_topic_drift_flag`

### Three-Model VAD Comparison

Open `notebooks/simulation_vad_model_comparison_workbench.ipynb` for one workflow
comparing **current BERT, NRC VAD v2.1, and MEmoLon MTL_grouped** on both earlier
test sets: 20 synthetic English context pairs and 50 saved news items across
SSSS, CCCC, PPPP, DDDD, CCPP, and PPCC (1,200 steps). No rewrites or LLM
evaluations are regenerated. The production VAD scorer and STDI formula remain
unchanged.

Cache MEmoLon and NRC resources before the first execution:

```bash
uv run python scripts/download_vad_comparison_resources.py
```

The MEmoLon English resource is cached as
`.cache/vad_lexicon_comparison/memolon_en.tsv`. NRC v2.1 is cached in `.cache/nrc_vad/`.
Resource files are ignored by Git. MEmoLon uses CC BY 4.0; NRC remains subject
to the [author's terms](https://saifmohammad.com/WebPages/nrc-vad.html).

Current `RobroKools/vad-bert` weights must already be cached locally. Both
lexical comparisons share the same current BERT revision and text-score cache.
The manifests record resource/input hashes, model revision, runtime, and
truncated-text counts. BERT uses the production 512-WordPiece limit; both
lexicons process full texts. No new dependencies are required.

Each lexicon averages the longest non-overlapping recognized term occurrences.
Unknown terms are excluded; no matches yield missing scores. MEmoLon averages
duplicate normalized entries. There is no negation handling or stopword
filtering. Normalize to the common nominal 0-1 scale with `(bert_score - 1)/4`,
`(nrc_native + 1)/2`, and `(memolon_native - 1)/8`. Native and common 1-5 scores
are also retained. Shared theoretical bounds do not calibrate the estimators.

The reusable orchestration is
`misinformation_simulation.analysis.three_model_vad_comparison`; it reuses the
existing lexical scorers and pairwise audit/formula checks. All joint summaries
use the same observations available for all three estimators. Individual scores
and missing values remain in the joint tables. Pairwise intermediate exports
are temporary and removed automatically after consolidation.

Current exports:

- `output/audit/VADThreeModelContextContrastAudit/`: all text scores, individual
  pair deltas, a three-model dimension summary, manifest, and report.
- `output/audit/SimulationVADModelComparison/`: side-by-side scores/drift/STDI,
  three-model chain/step summaries, inventory, manifest, report, and
  `comparison_dashboard.html`. The final notebook cell exports all interactive
  charts for both test sets into this self-contained HTML, with Plotly included
  for offline viewing. Keep the HTML in Git so visual results remain available
  when notebook outputs are cleared before committing.

Only the unified comparison notebook and its current results are retained for
this workflow. Earlier comparison notebooks, duplicate exports, and historical output
copies have been removed. The initial bilingual coverage pilot is retained
separately, as documented below. The original NRC comparison remains accessible in Git
at commit `d09cd838edbd827750933181775efe315c55763f`; that run used historical
BERT scores, while the unified comparison uses the current BERT revision.

Set `CHAIN_CODES`, `SELECTED_CHAIN_CODE`, `SELECTED_NEWS_ID`, and
`SELECTED_PAIR_ID` in the unified notebook to adjust scope and inspect examples.
Hypothetical STDI replaces only VAD after validating the historical formula
and saved BERT drift, preserving non-VAD contributions. Repeated news/steps are
dependent; cumulative STDI can exceed one. These English-only comparisons lack
independent human VAD ratings. Coverage, variation, or proximity to BERT do not
establish accuracy, and the results do not validate Portuguese.

### Initial Portuguese and English VAD Pilot

The initial bilingual diagnostic is retained in
`output/audit/PortugueseVADLexiconComparison/`, separately from the unified
English simulation/context comparison. It compares NRC v1, MEmoLon MTL_grouped,
NRC v2.1, and the current BERT on 35 fixed samples: three Portuguese local-news
captions, their manual English translations, one coverage-only TSE headline,
and affect-word/negation controls. NRC v2.1 and BERT Portuguese results are
unsupported-language probes; lexical token coverage is not applicable to BERT.

The report is `findings.md`; `scores.csv`, `matched_terms.csv`,
`news_coverage.csv`, `model_deltas.csv`, `control_deltas.csv`,
`control_summary.csv`, and `translation_deltas.csv` retain the detailed results.
`manifest.json` records the full samples, source URLs, resource/model hashes,
runtime, and scoring method. `recovery_manifest.json` records restoration from
the retained cache and comparison against the previous results. These collected
captions and manual translations are diagnostic inputs, without independent
human document VAD labels or verification of their factual claims.

```bash
uv run python scripts/download_vad_comparison_resources.py --include-portuguese-pilot
uv run python scripts/compare_portuguese_vad_lexicons.py
```

The optional download flag caches NRC v1 and Portuguese MEmoLon alongside
the resources used by the unified notebook. Pilot download metadata is stored
separately, so preparing the English comparison does not overwrite it. The
scoring script runs offline once the resources, BERT snapshot, and original
`output/elections_2026/simulation_candidates.csv` are cached.

### Fake vs True VAD Analysis

Notebook:

- [notebooks/fake_true_vad_workbench.ipynb](notebooks/fake_true_vad_workbench.ipynb)

It loads `data/FakeVsTrueVAD`, applies the reusable VAD module from `src`, and exports:

- row-level VAD scores for each article
- `fake` vs `true` summary tables
- subject-level summaries
- high/low examples by VAD dimension

Primary sources:

- Hugging Face model card: [https://huggingface.co/RobroKools/vad-bert](https://huggingface.co/RobroKools/vad-bert)
- Model dataset reference: [https://huggingface.co/datasets/reallycarlaost/emobank](https://huggingface.co/datasets/reallycarlaost/emobank)

By default the project uses the Hugging Face model `RobroKools/vad-bert`, loaded through `transformers.AutoModelForSequenceClassification`. The first run may download model weights from Hugging Face.

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
			TopicDriftAudit/
				*_stdi_audit.csv
				all_datasets_stdi_audit.csv
				stdi_summary.csv
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
- `output/runs/20260322_185309/audit/TopicDriftAudit/all_datasets_stdi_audit.csv`
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
| `original_*` | `annotate_stdi_for_rewrites` / `annotate_stdi_for_version_chain` | Structured topic extraction for the original article |
| `<version>_*` | `annotate_stdi_for_rewrites` / `annotate_stdi_for_version_chain` | Structured topic extraction and STDI metrics for each rewritten version |

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

### 5) Columns created in STDI audit (`topic_drift_audit_workbench.ipynb`)

| Column | Created by | Meaning |
| --- | --- | --- |
| `original_main_topic` | `annotate_stdi_for_rewrites` | Extracted primary topic of the original article |
| `original_subtopics` | `annotate_stdi_for_rewrites` | JSON array of extracted original subtopics |
| `original_central_entities` | `annotate_stdi_for_rewrites` | JSON array of central original entities |
| `original_central_relations` | `annotate_stdi_for_rewrites` | JSON array of original `(subject, action, object)` relations |
| `original_has_internal_contradiction` | `annotate_stdi_for_rewrites` | Whether the original text contains internal contradictions |
| `original_internal_contradiction_score` | `annotate_stdi_for_rewrites` | Severity/centrality of internal contradiction in the original text |
| `rewritten_news_main_topic` | `annotate_stdi_for_rewrites` | Extracted primary topic of the rewritten article |
| `rewritten_news_stdi_vs_original` | `annotate_stdi_for_rewrites` | Final STDI score against the original article |
| `rewritten_news_theme_drift_vs_original` | `annotate_stdi_for_rewrites` | Binary lexical main-topic drift component (graph embedding scores can be continuous) |
| `rewritten_news_subtopic_drift_vs_original` | `annotate_stdi_for_rewrites` | Subtopic drift component |
| `rewritten_news_entity_drift_vs_original` | `annotate_stdi_for_rewrites` | Central-entity drift component |
| `rewritten_news_relation_drift_vs_original` | `annotate_stdi_for_rewrites` | Central-relation drift component |
| `rewritten_news_has_internal_contradiction` | `annotate_stdi_for_rewrites` | Whether the rewritten text contains internal contradictions |
| `rewritten_news_internal_contradiction_score` | `annotate_stdi_for_rewrites` | Severity/centrality of internal contradiction in the rewritten text |
| `rewritten_news_contradiction_drift_vs_original` | `annotate_stdi_for_rewrites` | Rewritten text's internal-contradiction score, not a difference from the original |
| `rewritten_news_contradiction_drift_incremental` | `annotate_stdi_for_rewrites` | Same rewritten text's internal-contradiction score, used in incremental STDI |
| `high_topic_drift_flag` | STDI notebook | Whether STDI is above the configured threshold |

### 6) Summary CSV columns

| File | Key columns |
| --- | --- |
| `audit_summary.csv` | `dataset_name`, `source_file`, `rows`, `suspects`, `suspect_rate` |
| `stdi_summary.csv` | `dataset_name`, `source_file`, `rows`, `rows_with_successful_stdi`, `high_drift_count`, `high_drift_rate`, `mean_stdi`, `max_stdi` |
| `pretrained_fake_news_summary.csv` | `dataset_name`, `source_file`, `rows`, `fake_rate` |

### 7) Metadata row in fetched CSVs

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

### Model provenance

New graph runs record `topic_drift_model` and `topic_drift_provider` in the summary.
Each step retains its rewrite `model` and `provider`, and adds `metadata_rewrite_model`,
`metadata_rewrite_provider`, `metadata_topic_drift_model`, `metadata_topic_drift_provider`,
`metadata_vad_model`, and `metadata_stdi_embedding_model`. These identifiers describe
configured request models; operation statuses indicate whether each operation ran or
failed. They do not identify an API snapshot behind an alias. Lexical comparisons have
no embedding model. Custom scorers and embedders are identified explicitly.

Existing output files retain their historical model identifiers. Older runs lacking an
evaluation model must not be assigned the current default retroactively: that model is
unknown unless an original configuration or log establishes it. GPT-5.6 Luna remains
available for explicit selection. Resuming false-to-true rewriting preserves the model
and provider of reused successful rows, including missing provenance in older data.
Comparison outputs record models for newly requested extraction and semantic evaluation;
reused structures retain existing provenance instead of acquiring the current default.
