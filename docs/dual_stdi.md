# Dual STDI evaluation

Interaction-graph runs now default to `dual`. The **Dual STDI (embeddings + LLM judge)**
checkbox is enabled by default in execution settings. Uncheck it to run the structured
`cluster` branch alone, including polarity/numeric adjustments without judge requests.
`lexical` remains available through the Python API and CLI. The extraction
provider/model also evaluates the judge; rewriting nodes retain their own configurations.
Dual evaluation adds LLM requests, and an initial run may download the embedding/VAD models.

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
and `dual_stdi_v4`. Prompt versions invalidate request caches for new runs.
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

Both complete branches reuse the same VAD drift:

```text
C = (theme + subtopic + entity + relation) / 4
J = C + (1-C)*0.20*internal_contradiction
STDI_branch = J + (1-J)*0.20*VAD_drift
STDI_final = (STDI_embedding + STDI_llm_judge) / 2
method_gap = abs(STDI_embedding-STDI_llm_judge)
```

Full precision is retained until display. Optional qualifier diagnostics do not block
the branch. Failed extraction, unusable relation cores, incomplete VAD, invalid numeric judge
output and provider errors never become zero scores. Available complete
branches are retained, but the final mean requires both. Exact identical text/context reuses
the valid reference extraction and yields zero for both branches without a new judge call.
The method gap measures disagreement, not calibrated confidence.

## Inspecting results

Step details show both branch scores, disagreement, and expandable components/qualifiers.
The existing `stdi_vs_original` and `stdi_incremental` fields hold the final mean in dual mode.
Legacy category columns explicitly represent the embedding branch, not averaged components.
Both complete branch component sets appear in `metadata_dual_stdi_vs_original` and
`metadata_dual_stdi_incremental` in JSONL exports. These records include relation baselines,
adjustments, raw judge responses, rationale, input hashes and configuration versions.
Graph step schema 4 also exposes the evaluations separately as `cluster_evaluation_vs_original`,
`cluster_evaluation_incremental`, `llm_judge_evaluation_vs_original` and
`llm_judge_evaluation_incremental`. Each contains the complete branch result, including its
status, metrics, diagnostics and available error/provenance details.

All eleven metrics are exported as separate flat columns, for example
`stdi_cluster_vs_original`, `relation_drift_cluster_incremental`,
`stdi_llm_judge_vs_original` and `relation_drift_llm_judge_incremental`.
`stdi_status_{branch}_{suffix}` and `stdi_error_{branch}_{suffix}` distinguish missing or failed
evaluations from measured zeros. The existing `stdi_embedding_*` columns remain compatible aliases
for the Cluster STDI. `stdi_method_gap_*` and `stdi_status_*` still describe the final Dual result.

The result tables and charts show Cluster and LLM judge scores alongside Dual. Node, news and
category summaries aggregate each branch independently. The per-step details show separate
expandable evaluations for Cluster, LLM judge and the final Dual result. CSV downloads include
the flat component columns, while saved JSONL retains the complete nested evaluations.
Importing historical Dual JSONL derives the separate columns from its existing metadata without
rewriting files or rerunning models; unavailable historical branch accumulations remain absent.

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

The runner saves canonical comparisons, additional uncached judge repetitions, variability
statistics, coverage, method disagreements and qualifier-policy checks. Repeated judgments
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

This is software documentation. No thesis LaTeX or bibliography files require an Overleaf
update, and repository edits are not synchronized to Overleaf automatically.
