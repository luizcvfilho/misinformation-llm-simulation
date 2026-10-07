# Dual STDI evaluation

Interaction-graph runs now default to `dual`. The **Dual STDI (embeddings + LLM judge)**
checkbox is enabled by default in execution settings. Uncheck it to run the structured
`cluster` or `llm` alone. Cluster includes polarity/numeric adjustments without judge requests.
Select Model, LLM or Dual VAD independently; the default remains Model.
The [VAD/STDI selection matrix](vad_stdi_methods.md) defines all nine combinations,
including paired branch results and the final Dual aggregation.
`lexical` remains available through the Python API and CLI. The extraction
provider/model also evaluates the judge; rewriting nodes retain their own configurations.
Dual evaluation defaults to **three LLM judge evaluations per distinct pair**. In execution
settings, **Repeat LLM judge evaluation** is checked by default. Use **LLM judge evaluations
per pair** to choose any integer of at least two; uncheck repetition for exactly one evaluation.
Both controls are disabled for Cluster; they remain available for LLM-only STDI.
The selected settings apply to every graph in the execution queue.
Dual evaluation adds LLM requests, and an initial run may download the embedding/VAD models.

## Repeated judge evaluations

Each draw receives the same complete texts, extracted structures, model, endpoint and rubric.
Draws do not see other judgments or embedding scores. Extraction, embeddings and VAD are reused.
Repetition is distinct from retrying failed requests: every draw has its own existing bounded
retry budget. Scores and the final 50/50 branch weights remain unchanged.

For each draw, calculate all five component scores and its complete STDI with its selected VAD.
The LLM branch then averages the complete per-draw metrics at full precision. In particular,
its STDI is the mean of the individual STDIs, **not** STDI recomputed from mean components:
the contradiction adjustment is nonlinear. The final Dual score averages this LLM branch STDI
with the embedding branch STDI after applying the selected VAD to both structural evaluations.
When VAD is also Dual, the displayed branches use Model/LLM VAD respectively, while the
final Dual recomputes both complete scores with Dual VAD; see the exact formula below.

`llm_judge.samples` retains each numbered draw's status, cache hit, component scores,
rationales, raw response/provenance and complete metrics. `llm_judge.statistics` records
count, mean, sample standard deviation (`ddof=1`), minimum and maximum for each judged
component and STDI. A standard deviation requires two available values; otherwise it is null.
These are descriptive repeatability statistics, not calibrated confidence intervals.
The branch records `requested_repeats`, `executed_repeats`, `valid_repeats` and `aggregation`.
`executed_repeats` counts processed draws, including cached ones, rather than API attempts.
Graph summaries/step metadata identify `stdi_judge_repeats` and `stdi_judge_aggregation`;
pair manifests identify `judge_repeats` and `judge_aggregation`.

A complete LLM branch requires all requested draws and complete VAD. If any draw fails after
its retry budget, successful draws and their descriptive statistics remain available, but
the LLM branch and final Dual score are unavailable (`partial`, or `failed` if no draw succeeds).
Failures never enter the mean as zeros. Cached successes survive a restart; only missing draws
need new requests. Cancellation is checked between draws and retains completed sample records.
The identical-text shortcut still issues zero judge calls and explicitly records that shortcut.

Every draw has a separate durable cache entry. The first keeps the historical single-judge
key; subsequent keys also include a zero-based `repeat_index`. Increasing from one to three
draws adds two calls, while lowering the count uses only the selected prefix of saved draws.
Model, prompt, text, context or structure changes invalidate the corresponding keys.
`uncached_judge=True` bypasses all draw caches without replacing canonical saved judgments.
Saved historical scores are not recalculated automatically.

For uncached pairs, three draws cost approximately three times as much as one **in the judge
stage**, excluding retry differences and provider prompt-cache discounts. Rewriting and
extraction costs do not multiply. Comparisons against the original and against the previous
version can be distinct pairs; repeated identical pairs reuse the cache.

The default of three is a project budget choice, not a literature-validated sufficient count.
[Saha, Wagde and Kveton (2026), *LLM-as-a-Judge on a Budget*](https://proceedings.mlr.press/v300/saha26a.html)
studies repeated queries for mean-score estimation and allocating more queries to pairs with
higher estimated variance. This implementation uses a fixed user-selected count, not their
adaptive allocation algorithm. [Haldar and Hockenmaier (2025), *Rating Roulette*](https://aclanthology.org/2025.findings-emnlp.1361/)
documents variation in judges' scores across runs. Repetition describes and can reduce sampling
noise; it does not establish factual accuracy or agreement with independent human annotations.
Validate one versus multiple draws on a predefined human-reviewed sample before claiming gains.

Python configuration: `run_news_interaction_graph(..., stdi_judge_repeats=3)` and
`compare_dual_stdi(..., judge_repeats=3)`; `run_comparison_workflow(..., method="dual",
judge_repeats=3)` accepts the same setting. Use `1` to disable repetition. All reject nonpositive
or noninteger counts before requests. The low-level `compare_stdi_components_semantically`
function still performs one draw; legacy `llm_semantic` and manual annotation workflows keep
their existing single-draw behavior. This setting applies to the Dual judge branch.

```powershell
uv run python scripts/run_interaction_graph.py --input data/news.csv --graph-config data/graph.json --stdi-judge-repeats 3
uv run python scripts/run_topic_drift_comparison.py --method dual --input output/stdi_manual_evaluation/scored_stdi_pairs.csv --output-dir output/dual_stdi_pilot --judge-repeats 3
```

Replace the illustrative input/configuration paths with existing project files.

## Computation

Each distinct text/title/configuration is extracted once with schema version 2 in `dual`
and new `cluster` runs. The legacy prompt remains available for legacy extraction. Relations preserve the
signed action, base action, affirmed/negated polarity and scope, contextual numeric values, assertion
type. There is no `evidence` field in relation extraction or judge output, and no supporting
passage validation. Optional annotation fields are diagnostic metadata.
Intransitive actions may have an empty object. Purposes and attribution remain in action/object
and are preserved in the base action, rather than stored in a separate passage field.
Pure opinions/recommendations are recorded separately. Historical version-1 structures retain
their missing qualifiers and legacy comparison behavior; saved artifacts are not rewritten.

The current versions are `structured_extraction_v5`, `structured_judge_v4`, `cluster_v4`,
and `dual_stdi_v6`. Prompt versions invalidate request caches for new runs. Dual v5 introduced
repeated-judge aggregation; v6 identifies selectable VAD and final Dual recomputation.
The per-draw judge prompt and component STDI formula are unchanged.
The extraction prompt requires exactly `affirmed` or `negated`, never `unknown`.
Hypothesis/uncertainty is represented by assertion type, independently of polarity.
If a schema-2 response omits or violates the polarity enum, the parser normalizes it from
explicit negation in its signed action/scope, otherwise `affirmed`, and records a diagnostic.
This fallback concerns grammatical polarity, not factual truth or predicate equivalence.

The embedding branch retains the current theme, subtopic and entity comparisons, relation
core weights `0.45/0.25/0.15/0.15`, greedy one-to-one matching, and tie ordering. It compares
base actions with explicit negation and duration separated. Candidate alignment is independent
of qualifiers. Each aligned semantic distance `d_s` is adjusted as:

```text
delta_p = 0 for equal known polarity, 1 for different known polarity
d_adjusted = 0.80*d_s + 0.20*delta_p
d_value = min(abs(rewrite_value-reference_value)/abs(reference_value), 1)
d_numeric = max(distances of corresponding numeric slots)
d_relation = d_adjusted + (1-d_adjusted)*0.20*d_numeric
R = (sum(aligned distances) + number of unmatched relations) / max(list sizes)
```

Equal polarity can reduce drift even when actions/participants differ. There is no predicate
equivalence gate. Unmatched relations contribute 1 without a polarity discount. Exact
seconds/minutes/hours/days/weeks can be converted, including Portuguese units. Each relation's
`numeric_values` stores role, kind, value, unit, exact/approximate/unknown status and source expression.
Corresponding measurements are aligned by normalized role inside the already matched relation;
array position or numerical proximity never pairs unrelated values. Roles must be distinct and
stable; differences in role labels remain a possible source of extraction-driven drift.

Scalar values use the reference's absolute magnitude, supporting signed values and zero without
division errors. Two zeros yield zero; zero-to-nonzero yields one, with no defined percentage.
Explicit numerical addition/omission contributes one. The user selected **maximum**, not mean,
for multiple numerical changes, keeping a single fixed 0.20 adjustment per relation. Exact ranges
can be represented as distinct lower/upper bound roles. Approximate values and unspecified bounds
remain unavailable, rather than being invented or treated as measured zero.

Compatible elapsed-time, metric length/mass/volume, percent/fraction and age year/month units are
normalized. Other units require the same normalized label. Currencies require a shared currency
label; there is no exchange-rate conversion. Calendar months/years are not converted to seconds.
Complete ISO dates and identifiers use equality (zero if equal, one otherwise), with no arithmetic
percentage; standalone numerical years retain scalar comparison. Identifiers retain leading zeros.
Raw scalar percentage changes beyond the cap remain in the exported comparison details.

Missing, invalid, approximate, duplicate-role or incompatible measurements retain an unavailable
numeric adjustment and diagnostics. The branch keeps its polarity-adjusted semantic distance.
`numeric_values=None` distinguishes historical unavailable qualifiers from explicit absence `[]`.
When both structures predate the new numeric slots, the old duration comparator is retained;
those structures do not acquire inferred counts, amounts or other numerical annotations.
The legacy duration fields, `duration` details and `duration_weight` alias remain compatible.
New numeric slots include duration, and the legacy duration adjustment is not additionally applied.
Exports identify `numeric_weight=0.20` and `numeric_aggregation=maximum`.

The independent judge reads the full texts and shared structures, supplies all five component
scores on the anchored `0/0.25/0.5/0.75/1` rubric, and provides concise rationales.
Numeric score validation remains mandatory; missing rationales produce diagnostics.
Its relation score already includes semantic polarity, numeric changes, roles, purposes and omissions;
embedding adjustments are not applied again. The judge does not receive embedding scores or
persona labels and does not evaluate factual truth against external knowledge.

The final Dual calculation applies the selected VAD drift to both structural evaluations:

```text
C = (theme + subtopic + entity + relation) / 4
J = C + (1-C)*0.20*internal_contradiction
STDI_branch = J + (1-J)*0.20*VAD_drift
STDI_final = (STDI_embedding_with_selected_VAD + STDI_llm_judge_with_selected_VAD) / 2
method_gap = abs(displayed_STDI_embedding-displayed_STDI_llm_judge)
```

Model-only or LLM-only VAD is shared across the displayed branches. With Dual VAD and
Dual STDI, displayed Cluster uses Model VAD and displayed LLM uses LLM VAD; the final
score uses their mean drift as VAD in both structural evaluations before averaging.
It may therefore differ from the mean of the displayed paired branch scores.

Full precision is retained until display. Optional qualifier diagnostics do not block
the branch. Failed extraction, unusable relation cores, incomplete VAD, invalid numeric judge
output and provider errors never become zero scores. Available complete
branches are retained, but the final mean requires both. Exact identical text/context reuses
the valid reference extraction and yields zero for both branches without a new judge call.
The method gap measures disagreement, not calibrated confidence.

## Inspecting results

Step details show both branch scores, disagreement, and expandable components/qualifiers.
The existing `stdi_vs_original` and `stdi_incremental` fields hold the final mean in dual mode.
New main graph category columns represent the selected evaluation. Legacy Dual graph
category columns represent the embedding branch.
Both complete branch component sets appear in `metadata_dual_stdi_vs_original` and
`metadata_dual_stdi_incremental` in JSONL exports. These records include relation baselines,
adjustments, raw judge responses, rationale, input hashes and configuration versions.
Graph step schema 5 exposes the evaluations separately as `cluster_evaluation_vs_original`,
`cluster_evaluation_incremental`, `llm_judge_evaluation_vs_original` and
`llm_judge_evaluation_incremental`. Each contains the complete branch result, including its
status, metrics, diagnostics and available error/provenance details.

All eleven metrics are exported as separate flat columns, for example
`stdi_cluster_vs_original`, `relation_drift_cluster_incremental`,
`stdi_llm_judge_vs_original` and `relation_drift_llm_judge_incremental`.
`stdi_status_{branch}_{suffix}` and `stdi_error_{branch}_{suffix}` distinguish missing or failed
evaluations from measured zeros. The existing `stdi_embedding_*` columns remain compatible aliases
for the Cluster STDI. `stdi_method_gap_*` and `stdi_status_*` still describe the final Dual result.

The stored summaries can aggregate Cluster and LLM judge scores alongside Dual. Node, news and
category summaries aggregate each branch independently. The per-step details show separate
expandable evaluations for Cluster, LLM judge and the final Dual result. CSV downloads include
the flat component columns, while saved JSONL retains the complete nested evaluations.
The **STDI evaluation** selector in the Results dashboard and the STDI analysis sidebar switches
the existing charts, tables, component views, categories, persona/transition comparisons and cases
between **Dual**, **Cluster** and **LLM**, when those scores are available. Dual is the default when
available. A selected method never borrows another method's scores for missing evaluations.
Runs whose method was not recorded retain a **Saved evaluation (legacy)** option rather than
being labelled as Dual. Known standalone Cluster runs remain available as Cluster.

In the selected Dual view, new component values use explicitly saved final Dual metrics.
Historical files fall back to the mean of their shared-VAD branch metrics. The total STDI
remains the saved final score, rather than being recomputed from averaged components.
Branch views use their own complete
component sets, evaluation status and chain completeness. Historical branch cumulative scores are
reconstructed from saved increments within each news item and graph, with valid-step counts.
Unavailable increments remain missing; cumulative totals sum only available scores.
The selectors create an in-memory view and never overwrite persisted results or request new model
evaluations. Step CSV downloads identify the selected method in `stdi_evaluation`; analysis summary
downloads contain the selected method's scores.
Importing historical Dual JSONL derives the separate columns from its existing metadata without
rewriting files or rerunning models; unavailable historical increments remain missing.

The pair-comparison workflow similarly exports `cluster_*` and `llm_judge_*` metrics,
`cluster_evaluation_json` and `llm_judge_evaluation_json`, retaining `embedding_*` aliases and
the existing final `stdi` and `dual_evaluation_json` columns.

`stdi_cumulative` sums valid incremental final means and can exceed 1.
`stdi_cumulative_valid_steps` and `stdi_chain_complete` distinguish complete chains from
partial sums. Judge failures do not change a successfully rewritten text into a rewrite error.
`stdi_cluster_cumulative` and `stdi_llm_judge_cumulative` independently sum each branch's available
incremental scores. Each has its own `stdi_{branch}_cumulative_valid_steps` and
`stdi_{branch}_chain_complete`; a failed branch does not discard the other branch's valid score.
Accumulations reset for every news item and retain full precision.
Cancellation stops further judge calls; already evaluated results remain available.

Canonical extraction/judge outputs are stored under `evaluation_cache/` in the execution
directory. Queued graphs share the batch cache. The API accepts `stdi_cache_dir` to reuse a
chosen directory. Cache keys include texts, title, relevant structures, model/provider,
endpoint and prompt version. API keys are not stored in cache keys. Reusing saved outputs
makes numerical aggregation reproducible; new LLM calls can still vary.

## Evaluate saved pairs before a full simulation

```powershell
uv run python scripts/run_topic_drift_comparison.py --method dual --input output/stdi_manual_evaluation/scored_stdi_pairs.csv --output-dir output/dual_stdi_pilot --max-rows 3
```

This re-extracts historical version-1 structures with the extended schema and keeps the input
file intact. It writes `comparison_results.csv`, a manifest, and cached responses to a new
directory. Complete embedding/judge scores and both branch component sets are exported.
`historical_embedding_stdi` recomputes the old embedding formula on shared new structures;
it is a comparator baseline, not a claim to reproduce historical extraction outputs.
Previously saved columns (including `calculated_stdi`) are retained for that distinction.

```powershell
uv run python scripts/evaluate_dual_stdi.py --input data/synthetic/dual_stdi_validation_pairs.json --output-dir output/dual_stdi_development --split development --judge-repeats 3
uv run python scripts/evaluate_dual_stdi.py --input data/synthetic/dual_stdi_validation_pairs.json --output-dir output/dual_stdi_held_out --split held_out --judge-repeats 3
```

The runner defaults to three canonical draws per pair (`--judge-evaluations 3`). Its existing
`--judge-repeats` option counts **additional diagnostic single draws** on the selected sample,
not the number of canonical draws. These diagnostic draws remain unaggregated so that their
variability can be measured directly. The runner saves canonical comparisons, additional
uncached judge repetitions, variability statistics, coverage, disagreements and policy checks.
Repeated diagnostic judgments
do not replace the canonical cached judgment. The standalone comparison command also accepts
`--uncached-judge`. Human expected scores enter error statistics only when
`manual_review_status=reviewed`; retain disagreement and annotation history separately.

The new fixtures contain 12 development and 6 held-out diagnostic pairs with distinct story
families, English/Portuguese units, polarity inversions, paraphrase, actor reversal, uncertainty,
recommendation and internal contradiction. They are assistant-authored policy examples with
human review pending, not validated annotations. The existing 36 controlled pairs remain
unchanged and can also be supplied to the evaluation runner. Do not inspect held-out outputs
while refining development policies. Weights and the 50/50 mean stay fixed in both evaluations.

## Validation status and limits

Automated tests cover arithmetic bounds, both polarity directions, duration normalization and
edge cases, greedy normalization, unavailable means, precision, schema round trips, durable
caching, identity, graph failure handling and the UI toggle. They use frozen structures and
mocked model outputs and therefore do not establish extraction accuracy or empirical superiority.
Real-model judge repeatability, independent human annotation and held-out findings still require
running the supplied evaluation workflow and reviewing its evidence. Neither averaging nor the
selected 0.20 strengths have been empirically validated by these software tests.

This is software documentation. The repeatability sources are also recorded in the repository's
`references.bib` and thesis literature map. To use them in Overleaf, manually update
`elementos-postextuais/referencias.bib` from `references.bib`; no thesis LaTeX source was edited
for this implementation. Repository edits are not synchronized to Overleaf automatically.
