# Final implementation plan: dual STDI evaluation and arithmetic mean

Status: final architecture selected by the user on 2026-10-05; software implementation available; real-model and independent human validation pending. Latest user clarification: retain the current relation comparison model, make different polarity raise the distance and equal polarity lower it, remove predicate-equivalence gating, and fix both polarity and quantity/duration contribution strengths at 0.20. This replaces the earlier polarity-only increment and lexical-equivalence proposals. The original planning task changed the plan only. See docs/dual_stdi.md for the implemented software and remaining empirical validation.

## 1. Objective and selected architecture

Measure information and affective drift between a reference news text and its rewrite. Produce two complete STDI values and define the final value as their unweighted arithmetic mean:

    STDI_final = (STDI_embedding + STDI_llm_judge) / 2

The selected methods are:

- Embedding-based STDI: current semantic comparison, with a bidirectional binary polarity adjustment and a duration increment in relation comparison.
- LLM-as-a-Judge STDI: contextual judgments of theme, subtopics, entities, relations, and internal contradiction, converted to STDI by the existing numerical aggregation.
- Shared VAD: calculate valence, arousal, and dominance drift once for the pair and reuse those values in both methods.

The 50/50 mean is the user's selected aggregation policy. Validation measures its behavior and limitations; it does not automatically replace it with another weighting policy.

STDI measures transformation relative to a reference text. It does not establish factual truth, belief, exposure, or sharing behavior. Information absent from the original may be an unsupported addition without being false in the world.

## 2. Extraction and shared inputs

Retain the existing main topic, subtopics, central entities, subject/action/object relations, narrative frame, and internal-contradiction fields. Introduce a backward-compatible, versioned extraction extension with targeted additions:

- Explicit affirmed/negated polarity, the predicate it concerns, and negation scope.
- Duration value and unit, preserving the original duration expression.
- Supporting text spans and status for unknown or ambiguous qualifiers.
- Original signed predicate expression and a base action expression for separately scored qualifiers.

Adjust the prompt to preserve actors, actions, quantities, purposes, attribution, and uncertainty. Do not infer a speaker or event not supported by the text. Keep hypothesis, recommendation, and asserted action distinguishable without requiring a complete new proposition schema in the first release. Record pure opinion/recommendation separately from claims about what happened.

A source-event assertion and an added purpose must not be counted twice. Use the existing relation representation where possible; preserve supporting text for distinctions it cannot fully encode.

Extract each unique text/context once and reuse the saved structure across methods and chain comparisons. Both methods use the same text versions, title context, and saved structures. The judge also reads the full texts as primary evidence; its input is richer than an embedding comparison of extracted fields. Document this difference.

Existing records remain readable. Missing new fields in historical records mean unavailable, not affirmative polarity or zero duration. Keep historical scores unchanged.

## 3. Embedding-based relation comparison

Retain the current pair-level semantic calculation for participants, action, and object:

    s_core = 0.45*s_relation_text + 0.25*s_subject + 0.15*s_action + 0.15*s_object
    d_s = 1 - s_core

The existing TopicRelation holds subject, action, and object strings. A relation links its participants through an action; it is not just an untyped set of entities. Compare each field across text versions. The relation-text comparison embeds the entire labeled subject/action/object triple. Subject and object similarities use the existing entity index, and action similarity uses the action index. The similarities are normalized cosine values clipped to [0, 1], with exact normalized strings receiving similarity 1. Cluster labels are not a synonym or identity gate in this pair-level calculation.

The separate entity-drift component compares the inventory of entities. Reusing participants inside relation comparison captures their placement in subject/object roles and their association with an action. Retain this distinction and the current inner weights rather than replacing them with a new unvalidated three-way weighting.

Retain the current greedy one-to-one relation matching for the first implementation: calculate candidate core similarities, select the strongest available pairs, and avoid reusing a relation. Preserve fixed ordering and tie-breaking for reproducibility. Evaluate active/passive wording and role reversals; the current method does not guarantee that these cases are correctly aligned. Do not use polarity equality or duration equality as a prerequisite for candidate correspondence.

For a valid aligned relation pair, first adjust the semantic distance by binary polarity:

    delta_p = 0 if polarity_reference == polarity_rewrite else 1
    d_adjusted = (1 - lambda_p)*d_s + lambda_p*delta_p

Then add the duration contribution over the remaining distance:

    d_relation = d_adjusted + (1 - d_adjusted)*lambda_t*d_t

The user-selected contribution strengths are fixed:

    lambda_p = 0.20
    lambda_t = 0.20
    d_adjusted = 0.80*d_s + 0.20*delta_p
    d_relation = d_adjusted + (1 - d_adjusted)*0.20*d_t

Here lambda_t is the quantity/duration parameter, initially applied to the duration policy described below. Equal parameter values give equal configured maximum strength, not identical realized changes for every pair: polarity is binary, quantity is gradual, and duration acts on the remaining distance after polarity adjustment. These parameters are not direct percentages of the complete STDI.

All distances and lambda values lie in [0, 1]. For a nonsaturated semantic distance, different polarity raises it toward 1; equal polarity lowers it toward 0. With d_s=0.60, equal polarity yields 0.48 and different polarity yields 0.68 before duration. For a preserved core with d_s=0, an isolated polarity inversion gives 0.20; an isolated duration change from 10 to 11 gives 0.02; an isolated duration change from 10 to 20 gives 0.20. The selected strengths are methodological settings, not empirically validated weights.

This deliberately replaces the earlier guarantee that qualifiers could never lower the semantic baseline. Equal polarity can now lower the distance even if the action or participants changed. Record the semantic baseline, polarity adjustment, and final distance separately. A neutral duration factor leaves the polarity-adjusted value intact, with no weight renormalization.

Aggregate over valid aligned relations, additions, and omissions:

    R_embedding = (sum(d_relation for matched pairs) + unmatched_relations) /
                  max(reference_relation_count, rewrite_relation_count)

This preserves the existing maximum-list-size normalization for complete greedy matching. An unmatched valid relation contributes distance 1; do not apply a polarity discount to a relation with no counterpart. A qualifier omission does not imply that its event was omitted. Validate relation granularity, duplicates, and empty sets. Return zero for two empty relation sets only when successful extraction establishes absence. Incomplete required fields do not disappear from the denominator: flag the comparison as incomplete.

## 4. Binary polarity and duration rules

The embedding branch compares the two extracted polarity labels directly, independently of whether the verbs are equivalent:

- Same known label: delta_p=0, reducing the semantic distance by the configured polarity weight.
- Different known labels: delta_p=1, increasing the semantic distance by the configured polarity weight.
- Missing or ambiguous extracted label: incomplete qualifier assessment; do not assume equality or invent an affirmative value.

Remove predicate-equivalence gates, synonym dictionaries, lexical-resource selection, and the earlier verb-pair-specific exceptions from this branch. Under the user's selected rule, both help/not-assist and ignore/not-help have different extracted grammatical polarity and therefore receive the increasing adjustment. The independent LLM judge remains responsible for contextual semantic judgment in its own branch; do not feed its score back into the embedding adjustment.

Extraction still needs to identify which action is affirmed or negated. This is qualifier extraction, not a later synonym-equivalence test. Separate explicit syntactic negation and duration from the action core when they are scored independently; preserve the original expressions as evidence. Do not rewrite a lexically negative verb into a different positive verb. Retain the current encoder and comparison method for the resulting subject/action/object core.

For exact comparable durations, normalize units and use the reference value:

    relative_change = abs(t_rewrite - t_reference) / t_reference
    percentage_change = 100*relative_change
    d_t = min(relative_change, 1)

Examples: 10 to 11 days gives 10% and d_t=0.10; 10 to 20 gives 100% and d_t=1; 20 to 10 gives 50% and d_t=0.50. One day and 24 hours give zero after conversion. Preserve the raw percentage beyond the cap.

Duration edge cases:

- Absent in both texts: neutral duration factor.
- Explicitly present in only one: flag addition/omission; provisional d_t=1, percentage not applicable.
- Both numeric values zero: d_t=0, percentage undefined.
- Reference zero and rewrite positive: d_t=1, percentage undefined.
- Unknown extraction, incompatible units, or ambiguous scope: incomplete assessment, not zero.

Start with exact convertible durations. Calendar months/years, intervals, approximations, and other numerical quantities require separately evaluated policies. Portuguese normalization requires its own fixtures; the existing controlled corpus is English.

## 5. LLM-as-a-Judge method

Adapt the existing semantic comparator. Supply reference and rewritten texts, title context, and shared structures. Ask for structured judgments of:

- Theme drift.
- Subtopic drift.
- Entity drift.
- Relation drift: actions, responsibility, roles, attributed purposes, semantic polarity, duration, additions, and omissions.
- Internal contradiction within the rewrite, distinguished from conflict between versions.

Use the existing anchored levels 0, 0.25, 0.5, 0.75, and 1 as the initial rubric: preserved meaning, slight change, relevant partial change, strong change, and essentially different. Require short justifications and supporting passages, including relation-level examples when relevant. These categories are a scoring convention, not a validated interval scale.

The judge must assess meaning rather than writing quality, ideological agreement, or truth against external knowledge. It must preserve distinctions between a hypothesis, an attributed intention, a recommendation, and an asserted action. Do not provide persona labels or embedding scores before judgment.

Judge relation drift already includes semantic polarity and duration. Do not apply the embedding branch's polarity/duration increments to this result again. The judge does not estimate VAD.

Save raw responses, parsed scores, rationale, evidence, model/provider configuration, effective supported sampling settings, prompt version, and input hashes. Use a fixed rubric and canonical cached judgment per pair. Repeated uncached judgments on an evaluation sample measure variability; caching does not make new model calls deterministic.

## 6. Complete STDI calculation and final mean

Both methods use the existing global weights and numerical formula:

    C_m = 0.25*(theme_m + subtopic_m + entity_m + relation_m)
    J_m = C_m + (1 - C_m)*0.2*contradiction_m
    STDI_m = J_m + (1 - J_m)*0.2*VAD_drift

Here m is either embedding or llm_judge. The embedding method retains its current theme/subtopic/entity comparison, uses the new relation distance, and retains extraction-derived internal contradiction. The judge method supplies its own five component judgments. Both reuse the same calculated VAD drift.

Then calculate:

    STDI_final = (STDI_embedding + STDI_llm_judge) / 2
    method_gap = abs(STDI_embedding - STDI_llm_judge)

For example, branch values 0.30 and 0.70 yield STDI_final=0.50 and method_gap=0.40. The gap is disagreement, not calibrated uncertainty or confidence.

Average the two complete STDI values after their contradiction and VAD contributions. Do not substitute an average of relation scores or component scores as the selected final aggregation.

Calculate at full available precision and round for display. If either complete method is invalid or unavailable, preserve the available branch result and report the final mean as unavailable. Do not silently replace the mean with one branch or insert zero for failure. A semantic-only partial fallback remains explicitly labeled and does not masquerade as the complete method.

## 7. Validation and calibration

Use the existing 36 controlled pairs for development. Add manually reviewed cases covering qualifier extraction, known equal/different polarity, actor reversal, paraphrase, additions/omissions, and duration magnitudes and units. Remove the prior mandatory synonym-equivalence-gate protocol. Evaluate the direct polarity rule and independently evaluate the judge's contextual assessment.

First test comparison with manually supplied structures; then test actual extraction against those references. Evaluate alignment, qualifier preservation, correct binary adjustment, false changes on paraphrases, missed changes, coverage, judge repeatability, and numeric invariants.

Reserve new stories and verb-pair families for held-out evaluation. Annotate before inspecting model scores; use a second human annotator where feasible and document disagreement. Human judgments assess informational change, not factual verification.

Compare and retain:

- Historical current STDI baseline.
- New embedding-plus-qualifier-adjustments STDI.
- LLM-as-a-Judge STDI.
- The selected final 50/50 mean.

Keep the user-selected polarity and quantity/duration strengths fixed at 0.20 each during implementation and evaluation. Refine rubric details and normalization on development data, then freeze them before held-out evaluation. Keep the selected 50/50 mean unchanged. Validation measures the behavior of these selected settings rather than silently tuning them to other weights. Report when averaging retains a paraphrase error or dilutes a real change; do not assume fusion must outperform both branches. The prior audit is diagnostic evidence, not validation of this new architecture.

Required mathematical checks: bounds [0,1], equal polarity lowers or preserves an already-zero core distance, different polarity raises or preserves an already-maximal core distance, zero polarity weight preserves the core, absent duration leaves the polarity-adjusted value intact, duration unit equivalence, zero handling, preserved greedy aggregation normalization, final mean between both branch values, and unavailable mean when a complete branch fails. Duration may subsequently raise a value reduced by equal polarity; test both stages separately. Treat exact identical text/context as an identity case: reuse its valid extraction and VAD, set both comparison branches to zero, and record the identity shortcut rather than issuing a fresh judge call. Do not infer identity from independently inconsistent extractions. Repeated frozen structures and judge outputs must reproduce aggregation within recorded numerical tolerance.

## 8. Implementation sequence and integration points

1. Define shared schema extension, scoring rubric, direct polarity adjustment, validity states, and reviewed fixtures.
2. Implement extraction/parser/cache changes with backward compatibility.
3. Retain deterministic greedy relation alignment; implement duration normalization, direct polarity adjustment, and duration increment.
4. Adapt the existing judge with evidence-backed component outputs and saved responses.
5. Calculate both complete STDI values, fixed final mean, and disagreement metadata.
6. Run development calibration and held-out evaluation.
7. Add an opt-in simulation mode and result inspection, then pilot saved rewrites before generating a full new run.

Expected files: topic_drift/models.py, config/prompts.py, topic_drift/extraction.py, new relation/qualifier modules, topic_drift/cluster_comparison.py, topic_drift/semantic_comparison.py, topic_drift/metrics.py, topic_drift/comparison_workflow.py, simulation/graph.py, result UI/export code, and substantive tests.

For each chain step, compare against the original and the previous version. Average complete branch scores for each pair; cumulative final drift is the sum of valid incremental final scores, not a single [0,1] distance. Report incomplete chains and coverage.

Expose both branch values and components, final mean, method gap, qualifier evidence, binary polarity flags and adjustments, and valid/partial/failed status. Record exact extraction, comparator, prompt, model, and formula versions. Preserve old execution artifacts; reprocessing creates separately labeled outputs.

## 9. Methodological basis and completion criteria

Sentence-BERT supports embedding comparison: https://aclanthology.org/D19-1410/. G-Eval is a precedent for structured LLM evaluation: https://aclanthology.org/2023.emnlp-main.153/. Zheng et al. document LLM-judge capabilities and biases in other evaluation tasks: https://arxiv.org/abs/2306.05685. No lexical synonym resource is required by the selected direct polarity rule.

These sources do not validate the STDI formula, increments, contribution strengths, or 50/50 mean. Those are design decisions of this study. Small development fixtures cannot establish general validity. The complete pipeline includes model-dependent extraction and judgment; only numerical computation over fixed saved outputs is deterministic.

Completion requires implemented and documented dual computation, correct fixed mean and failure handling, traceable outputs, compatibility checks, reviewed validation fixtures, reported judge variability, and held-out findings. This plan does not authorize claims of empirical superiority before evaluation.

## 10. Files and Overleaf synchronization for the original planning task

In the original planning task, only docs/stdi_structured_comparison_implementation_plan.md was changed. Production code, prompts, tests, datasets, historical results, thesis LaTeX, and references.bib are unchanged.

The changed file is repository implementation documentation in English. It has no required Overleaf destination; keep it in the repository. No Overleaf file, including main.tex or any file beyond main.tex, requires updating for this task. Local changes are not synchronized automatically.


## 11. Software implementation update

The `dual` comparison method is implemented and is the default in the graph API and CLI.
The graph UI enables its checkbox by default and selects the existing `cluster` method when
it is disabled. The software includes schema-v2 extraction, qualifier adjustments, a contextual
judge with evidence, shared VAD, complete-score averaging, validity states, traceable exports,
caching, and valid incremental-chain coverage. Legacy comparison modes remain available.

Automated verification uses frozen structures and mocked model responses. The evaluation runner
in scripts/evaluate_dual_stdi.py supports saved rewrites, held-out selection, uncached judge
repetitions and reports; data/synthetic/dual_stdi_validation_pairs.json contains additional
assistant-authored diagnostic cases with independent human review pending. Real-model variability
and held-out findings have not been established by implementation tests. Detailed usage and
limitations are documented in docs/dual_stdi.md. No thesis or bibliography file requires changes
in Overleaf, and local repository changes are not synchronized automatically.
