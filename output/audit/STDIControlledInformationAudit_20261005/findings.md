# Controlled audit of information changes in STDI: corrected rerun

## Finding

The corrected run produces complete scores for 36/36 primary pairs with cluster, 36/36 with the LLM judge, and 36/36 with both branches. All 42/42 saved extractions meet the current structured comparison requirements.

The cluster matches 6/9 diagnostic order checks, compared with 2/9 in the original 04 October audit. The judge matches 9/9 available checks. These expectations were authored for the controlled constructions; they are not human-validated accuracy targets.

## Material and execution

- The same three fictional English scenarios and twelve variants per scenario were reused: identity, paraphrase, normative opinion, tentative motive, asserted motive, opposite motive, negation, actor reassignment, quantity change, added action, omission, and internal contradiction.
- Fresh extraction was explicitly requested after the previous 05 October output directory was deleted. This run does not reuse the deleted responses. The original 04 October audit is preserved as the historical baseline.
- 36 distinct texts and 42 saved extractor responses, including six independent government extraction repetitions. The first original extraction is frozen across each scenario's variants.
- 48 cluster rows: 36 primary pairs, six direct motive contrasts, and six extraction-repeat comparisons. 48 rows have complete cluster scores.
- 42 judge rows: 36 primary pairs and six direct contrasts. 42 rows have complete LLM scores.
- Model/provider: `gpt-6-luna` / `chatgpt` for both extraction and judging. Extraction: `structured_extraction_v4` with schema 2; judge: `structured_judge_v3`; structured cluster: `cluster_v3`; dual version: `dual_stdi_v3`.
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`. VAD: `RobroKools/vad-bert`. Both local models were loaded from the existing cache; both branches reuse the same VAD scores.
- No separate title is supplied to extraction. Display titles are not evidence. No persona-generated rewrites, external factual verification, or human annotation enter this audit.

The current corrections remove relation/judge evidence fields and passage validation, allow an empty object for intransitive actions, normalize missing grammatical polarity with diagnostics, and preserve semantic comparison when optional duration adjustment is unavailable. Usable relation cores, VAD and numeric scores remain required. The audit runner was also changed to use the current default retry budget instead of overriding it with one attempt.

The retry budget allows an initial attempt plus three retries for generation or parsing failures. Counts above refer to finalized saved responses, not total API attempts; the exact attempt count is not exported. Temperature is omitted for this model family, so the requested 0.1 is not an effective sampling setting.

Each content component retains weight 0.25. Internal contradiction and VAD each contribute 0.20 of the remaining distance. Explicit polarity and duration strengths are 0.20. The supplementary dual score is the 50/50 mean of the two complete branches. The judge receives full texts and shared structures; cluster receives the extracted representation. Their input information differs, and neither method is a human reference.

## Main comparison

All values are final STDI including VAD. The original and corrected audits use the same input corpus. New stochastic extractions, new prompts and comparator changes prevent attributing every difference to one correction alone.

| Scenario and change | Original cluster | Corrected cluster | Original LLM | Corrected LLM |
| --- | ---: | ---: | ---: | ---: |
| Government: paraphrase | 0.215154 | 0.056849 | 0.001420 | 0.001420 |
| Government: negation | 0.050166 | 0.104532 | 0.439332 | 0.439332 |
| Government: actor reassignment | 0.221769 | 0.121160 | 0.187926 | 0.187926 |
| Government: quantity change | 0.121229 | 0.136528 | 0.250350 | 0.250350 |
| Health: paraphrase | 0.172338 | 0.097066 | 0.003036 | 0.003036 |
| Health: negation | 0.171092 | 0.179029 | 0.438610 | 0.438610 |
| Health: actor reassignment | 0.196562 | 0.195445 | 0.313116 | 0.313116 |
| Health: quantity change | 0.113062 | 0.128976 | 0.250107 | 0.250107 |
| Election: paraphrase | 0.356553 | 0.211505 | 0.000692 | 0.000692 |
| Election: negation | 0.244031 | 0.202370 | 0.438438 | 0.438438 |
| Election: actor reassignment | 0.135889 | 0.124162 | 0.313077 | 0.375525 |
| Election: quantity change | 0.064627 | 0.127704 | 0.063041 | 0.187969 |

## Results by controlled change

Means below use only pairs for which both branches are complete; coverage is explicit.

| Change | Complete paired scenarios | Cluster mean | LLM mean |
| --- | ---: | ---: | ---: |
| identity | 3/3 | 0.000000 | 0.000000 |
| paraphrase | 3/3 | 0.121806 | 0.001716 |
| normative_opinion | 3/3 | 0.338603 | 0.295893 |
| motive_tentative | 3/3 | 0.244480 | 0.294853 |
| motive_asserted | 3/3 | 0.188664 | 0.316051 |
| opposite_motive | 3/3 | 0.194650 | 0.315473 |
| negation | 3/3 | 0.161977 | 0.438794 |
| actor_reassignment | 3/3 | 0.146923 | 0.292189 |
| quantity_change | 3/3 | 0.131069 | 0.229475 |
| new_action | 3/3 | 0.342642 | 0.399909 |
| omission | 3/3 | 0.454647 | 0.338099 |
| internal_contradiction | 3/3 | 0.395553 | 0.484206 |

## Paraphrase and extraction variability

| Scenario | Theme drift | Subtopic drift | Entity drift | Relation drift | Cluster STDI | LLM STDI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| government | 0.141806 | 0.000000 | 0.000000 | 0.080222 | 0.056849 | 0.001420 |
| health | 0.043650 | 0.276880 | 0.000000 | 0.056735 | 0.097066 | 0.003036 |
| election | 0.026398 | 0.456327 | 0.333333 | 0.027774 | 0.211505 | 0.000692 |

The election paraphrase omits `voter register` from the extracted central-entity list, although it remains in the input text. Entity drift is 0.333333 and subtopic drift is 0.456327. These representation differences contribute to its relatively high cluster score. The cluster's three failed ordering checks are in this scenario: negation, actor reassignment and quantity change all score below paraphrase.

The three primary identical-text controls reuse the frozen original structure and score zero. Independent extraction repetitions deliberately compare separately generated structures for the same text:

| Repetition | Cluster STDI |
| --- | ---: |
| government_identity_replicate_2 | 0.045228 |
| government_identity_replicate_3 | 0.064319 |
| government_motive_asserted_replicate_2 | 0.271583 |
| government_motive_asserted_replicate_3 | 0.159091 |
| government_negation_replicate_2 | 0.179156 |
| government_negation_replicate_3 | 0.173891 |

Nonzero identical-text repetition scores diagnose extractor variability. They must not be confused with the production identity shortcut, which reuses the original structure and returns zero.

## Polarity, duration, motives and opinions

For the two-day to twenty-day variants, the structured duration calculation retains the raw 900% relative change and caps duration distance at 1 before applying the 0.20 contribution. The election change from 120 to 1200 names is a count, not a duration; it does not receive that duration adjustment. Inspect the aligned relation details before interpreting its full score as sensitivity to the count itself.

The government asserted and opposite motives are preserved in `base_action`, including the purposes to weaken or protect the union. All six direct contrasts can now be compared by both branches:

| Direct contrast | Cluster STDI | LLM STDI |
| --- | ---: | ---: |
| government_motive_tentative_vs_motive_asserted | 0.128892 | 0.190019 |
| government_motive_asserted_vs_opposite_motive | 0.098229 | 0.375240 |
| health_motive_tentative_vs_motive_asserted | 0.054810 | 0.252700 |
| health_motive_asserted_vs_opposite_motive | 0.234948 | 0.377640 |
| election_motive_tentative_vs_motive_asserted | 0.320242 | 0.252002 |
| election_motive_asserted_vs_opposite_motive | 0.124745 | 0.376348 |

All three normative recommendations appear in `opinions`, but also occur as recommendation relations. The added unmatched recommendation relations contribute to relation drift. These are normative statements, not newly asserted past events; technical validity of an extraction does not establish conceptual accuracy of every component interpretation.

## Artifacts and reproduction

The output retains the previous format: `audit_review.html`, `score_comparison.png`, `scored_examples.md`, `scored_pairs.csv`, `semantic_method_comparison.csv`, `change_type_summary.csv`, `diagnostic_order_checks.csv`, `pairs_for_review.md`, saved responses and configuration/manifests. `historical_method_comparison.csv` compares the original audit with the corrected run and includes a legacy-comparator baseline on the new shared structures. `extraction_validation.csv` records extraction coverage.

The nine manual comparator probes retain version-1 relations and legacy `cluster_v2`; they provide continuity with the original audit, not validation of the new structured qualifiers.

```powershell
uv run python scripts/audit_stdi_information_changes.py --structured --stage all --output-dir output/audit/STDIControlledInformationAudit_20261005
uv run python scripts/audit_stdi_information_changes.py --structured --stage semantic --output-dir output/audit/STDIControlledInformationAudit_20261005
uv run python scripts/report_stdi_information_audit.py --output-dir output/audit/STDIControlledInformationAudit_20261005 --baseline-dir output/audit/STDIControlledInformationAudit_20261004 --export-png
```

The commands reuse this run's saved checkpoints. Use a new directory for another fresh extraction. Local execution used the existing uv environment and a workspace cache/temp directory. Source and checkpoint hashes are preserved in `evidence_manifest.json`.

## Validation, limits and Overleaf synchronization

Targeted flexible-validation, evidence-removal, retry and audit-runner tests passed. The new runner integration test confirms that invalid JSON is retried before a valid extraction checkpoint is saved. Artifact integrity checks cover coverage, bounds, identity controls, dual arithmetic, frozen originals and historical text equality.

This diagnostic run concerns three short synthetic scenarios and one extractor/judge model. The ordering checks and fixed weights have not been validated against independent human judgments. Informational drift, internal contradiction and external factual veracity remain distinct. The audit does not measure belief, exposure or sharing behavior.

Only audit-runner/test files and local output artifacts were changed. No thesis LaTeX, institutional template or bibliography file was edited. No Overleaf file requires an update, including `main.tex`; no file beyond `main.tex` needs changes. Local changes are not synchronized to Overleaf automatically.
