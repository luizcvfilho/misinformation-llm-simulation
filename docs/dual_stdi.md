# Dual STDI evaluation

Interaction-graph runs now default to `dual`. The **Dual STDI (embeddings + LLM judge)**
checkbox is enabled by default in execution settings. Uncheck it to run the existing
`cluster` method. `lexical` remains available through the Python API and CLI. The extraction
provider/model also evaluates the judge; rewriting nodes retain their own configurations.
Dual evaluation adds LLM requests, and an initial run may download the embedding/VAD models.

## Computation

Each distinct text/title/configuration is extracted once with schema version 2. Legacy
extraction prompts remain unchanged for legacy methods. Version-2 relations preserve the
signed action, base action, affirmed/negated polarity and scope, exact duration, assertion
type, and verbatim evidence. Pure opinions/recommendations are recorded separately.
Historical missing qualifiers remain unknown; existing execution artifacts are not rewritten.

The embedding branch retains the current theme, subtopic and entity comparisons, relation
core weights `0.45/0.25/0.15/0.15`, greedy one-to-one matching, and tie ordering. It compares
base actions with explicit negation and duration separated. Candidate alignment is independent
of qualifiers. Each aligned semantic distance `d_s` is adjusted as:

```text
delta_p = 0 for equal known polarity, 1 for different known polarity
d_adjusted = 0.80*d_s + 0.20*delta_p
d_t = min(abs(rewrite_duration-reference_duration)/reference_duration, 1)
d_relation = d_adjusted + (1-d_adjusted)*0.20*d_t
R = (sum(aligned distances) + number of unmatched relations) / max(list sizes)
```

Equal polarity can reduce drift even when actions/participants differ. There is no predicate
equivalence gate. Unmatched relations contribute 1 without a polarity discount. Exact
seconds/minutes/hours/days/weeks can be converted, including Portuguese units. Duration absent
in both texts is neutral; explicit addition/omission provisionally contributes `d_t=1`.
Two zeros produce zero distance, and zero-to-positive produces one; neither has a defined
percentage. Calendar months/years, ranges, approximations, incompatible units and unknown
scope remain incomplete. Raw percentages beyond the cap are retained in evidence details.

The independent judge reads the full texts and shared structures, supplies all five component
scores on the anchored `0/0.25/0.5/0.75/1` rubric, and provides rationale plus verbatim evidence.
Its relation score already includes semantic polarity, duration, roles, purposes and omissions;
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

Full precision is retained until display. Missing qualifiers, extraction failures, incomplete
VAD, invalid judge output and provider errors never become zero scores. Available complete
branches are retained, but the final mean requires both. Exact identical text/context reuses
the valid reference extraction and yields zero for both branches without a new judge call.
The method gap measures disagreement, not calibrated confidence.

## Inspecting results

Step details show both branch scores, disagreement, and expandable components/evidence.
The existing `stdi_vs_original` and `stdi_incremental` fields hold the final mean in dual mode.
Legacy category columns explicitly represent the embedding branch, not averaged components.
Both complete branch component sets appear in `metadata_dual_stdi_vs_original` and
`metadata_dual_stdi_incremental` in JSONL exports. These records include relation baselines,
adjustments, raw judge responses, rationale, passages, input hashes and configuration versions.
`stdi_embedding_*`, `stdi_llm_judge_*`, `stdi_method_gap_*` and `stdi_status_*` expose the summary.

`stdi_cumulative` sums valid incremental final means and can exceed 1.
`stdi_cumulative_valid_steps` and `stdi_chain_complete` distinguish complete chains from
partial sums. Judge failures do not change a successfully rewritten text into a rewrite error.
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
file intact. It writes `comparison_results.csv`, a manifest, and cached evidence to a new
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
