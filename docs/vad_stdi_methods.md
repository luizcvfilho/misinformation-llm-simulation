# Selecting VAD and STDI methods

New graph and pair-comparison runs select VAD independently from structural STDI evaluation.
The default remains `vad_method="model"` (`RobroKools/vad-bert`). Graph STDI defaults to
`dual`. Existing saved runs are not recalculated or assigned missing LLM ratings.

## Selection matrix

| VAD selection | STDI selection | Cluster result uses | LLM result uses | Dual result uses |
| --- | --- | --- | --- | --- |
| Model | Cluster | Model VAD | Not requested | Not requested |
| Model | LLM | Not requested | Model VAD | Not requested |
| Model | Dual | Model VAD | Model VAD | Model VAD |
| LLM | Cluster | LLM VAD | Not requested | Not requested |
| LLM | LLM | Not requested | LLM VAD | Not requested |
| LLM | Dual | LLM VAD | LLM VAD | LLM VAD |
| Dual | Cluster | Dual VAD | Not requested | Not requested |
| Dual | LLM | Not requested | Dual VAD | Not requested |
| Dual | Dual | Model VAD | LLM VAD | Dual VAD |

Selecting VAD Dual exports all three affective drifts, even when only one structural method
is requested. Selecting one VAD estimator evaluates that estimator alone. STDI Cluster still
requires LLM extraction, but does not request structural judge evaluations. STDI LLM alone
does not fit the embedding comparator. VAD LLM calls are separate from structural judge calls.

## VAD calculation and rubric

Each estimator rates the original and rewritten text independently on the nominal 1–5 scale.
For each dimension, calculate `min(abs(rewrite-reference)/4, 1)`, then average the three
dimension drifts. VAD Dual averages the resulting Model and LLM drifts, including the three
dimension drifts. It does not average raw vectors before calculating distance: opposite signed
changes must not cancel. There is no raw Dual VAD vector.

The LLM receives one complete text and the frozen `llm_vad_expressed_tone_v1` rubric.
It receives no paired text, persona, BERT rating or structural judge rating. It evaluates
expressed affective tone, returns three numeric scores, concise rationales, supporting excerpts
and ambiguities. This does not measure reader emotion, behavioral impact or factual truth.
See [the rubric and pilot protocol](llm_vad_comparison.md#rubric-and-comparison).
The current implementation evaluates full texts; passage-level evaluation remains future work.

VAD is evaluated once per exact text within a run. Durable LLM cache keys include the text,
model, provider, endpoint, rubric and system instruction. Model ratings have a separate cache
key with text, model name, maximum length and evaluation version. Injected test scorers bypass
durable caching. The LLM's raw responses, rationales and excerpts remain in saved evaluation
records. Structural judge repetitions reuse the same VAD ratings and do not multiply VAD calls.

## Final Dual STDI

The existing component formula and 0.20 strengths remain unchanged. Let `F(S,A)` denote the
complete STDI of structural evaluation `S` with affective drift `A`:

```text
C = (theme + subtopic + entity + relation) / 4
J = C + (1-C)*0.20*internal_contradiction
F(S,A) = J + (1-J)*0.20*A
A_dual = (A_model + A_llm) / 2
STDI_dual = (F(S_cluster,A_selected) + mean_j F(S_llm_j,A_selected)) / 2
```

When both selections are Dual, the displayed Cluster score is `F(S_cluster,A_model)` and
the displayed LLM score is `mean_j F(S_llm_j,A_llm)`. The final Dual score recalculates both
structural branches with `A_dual` before taking their 50/50 mean. Therefore it can differ
from the arithmetic mean of the two displayed paired scores. This makes the final Dual
result actually use Dual VAD. With Model-only or LLM-only VAD, the final value is also the
mean of the displayed branch scores. Repeated judgments average complete scores, preserving
the existing nonlinear contradiction adjustment.

The saved `method_gap` compares the displayed complete branch scores; it can include both
structural and affective disagreement when both selections are Dual. It is descriptive
disagreement, not calibrated confidence. The VAD record also exposes its own drift gap.

## Failure and output policy

Missing dimensions and failed ratings remain unavailable. Dual VAD requires both complete
estimators. With both selections Dual, one valid estimator can still support its paired
complete STDI branch, while the other branch and final Dual remain unavailable. A single
structural evaluation using Dual VAD requires both estimators. No failure enters a mean as zero.

Graph schema 5 saves individual text evaluations and original/incremental pair evaluations.
CSV columns include `vad_drift_model_vs_original`, `vad_drift_llm_incremental`,
`vad_drift_dual_incremental`, per-dimension drifts and statuses. Raw scores are available as
`vad_model_original_valence` and `vad_llm_rewritten_arousal`, for example.
`stdi_vad_source_cluster_*`, `stdi_vad_source_llm_judge_*` and `stdi_vad_source_dual_*`
identify the VAD source used by each result. Explicit `*_dual_*` component columns describe
the final Dual evaluation; dashboard projections use them instead of averaging the displayed
paired branch components. Main graph component columns describe the selected STDI evaluation.

Standalone Dual comparisons retain legacy unprefixed component columns as Cluster aliases;
`dual_*` columns contain the final Dual metrics and `stdi` contains the final score.
Nested evaluation JSON includes the complete records. `dual_stdi_v6` identifies the new
selection/aggregation behavior. Historical versions remain readable without new requests.

## Configuration

In execution settings, select **VAD method** independently. For LLM or Dual, set **VAD evaluator
model** and **VAD evaluator provider**. The initial defaults follow the extraction evaluator.
Keep **Dual STDI** enabled for all three STDI results, or disable it and choose **STDI method**
Cluster or LLM. Repetition controls apply to Dual and LLM structural evaluation.

Python APIs accept `vad_method`, `vad_llm_model` and `vad_llm_provider`; graph runs and
`run_comparison_workflow` also accept separate VAD credentials/endpoint and injected scorers.
When providers match, omitted VAD credentials/endpoint reuse the extraction configuration;
with a different provider, VAD uses that provider's environment defaults unless overridden.
Graph STDI uses `stdi_comparison_method="llm"` for the single LLM branch; the reusable pair
workflow retains `method="llm_semantic"` for its existing single-judge structural workflow.

```powershell
uv run python scripts/run_interaction_graph.py --input data/news.csv --graph-config data/graph.json --vad-method dual --stdi-comparison-method dual
uv run python scripts/run_topic_drift_comparison.py --input output/stdi_manual_evaluation/scored_stdi_pairs.csv --output-dir output/vad_dual_comparison --method dual --vad-method dual
```

Replace illustrative paths with existing input files. New LLM ratings incur API requests;
the default Model selection adds no VAD API calls. Automated validation uses fixed ratings
and judgments across all nine combinations, persistence, cache reuse, partial results and UI
selection. These checks establish implementation behavior, not superiority of an estimator.
