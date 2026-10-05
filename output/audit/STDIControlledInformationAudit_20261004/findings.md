# Controlled audit of information changes in STDI

## Finding

The current pipeline underreacts to several explicit changes in meaning while reacting strongly to some meaning-preserving paraphrases. This audit provides concrete evidence of both extraction variability and comparator limitations. The issue is not explained by the rewrite personas alone, and increasing the relation weight would not address the comparator's very low distances for some opposing claims.

This is a diagnostic study with fictional examples, not human validation of STDI. No production prompt, extractor, comparator, weight, or historical simulation score was changed.

## Material and execution

- Three short fictional English news scenarios: government/teachers-union negotiations, clinic closure, and voter-register revision.
- Twelve variants per scenario: identity, paraphrase, normative opinion, tentative motive, asserted motive, opposite motive, negation, actor reassignment, quantity change, added action, omission, and internal contradiction.
- 36 original/rewrite pairs, plus six direct comparisons between motive variants.
- 42 actual extraction responses: one per distinct text, plus six additional independent extractions of three government texts. Each original's first extraction is frozen across its variants.
- 48 scored comparisons using the current `cluster_v2` pipeline, including six extraction-repeat comparisons.
- Nine hand-specified structural probes isolate the comparator from the LLM extractor. These are controlled input structures, not substitute extraction outputs or human annotations.
- 42 actual responses from the project's existing optional LLM semantic comparator, evaluated on the 36 main pairs and six direct contrasts.
- Extractor and optional evaluator: `gpt-6-luna` through the configured `chatgpt` provider. Encoder: `sentence-transformers/all-MiniLM-L6-v2`. VAD: `RobroKools/vad-bert`, loaded from the existing local cache.
- No title is supplied to extraction. Display titles are not additional evidence. This aligns with the current repository's title-context change recorded at commit `35476c8`.
- Texts are authored controlled examples; there are no persona-generated rewrites in this experiment. This avoids uncontrolled changes during pair construction.

Every response is saved separately. The complete extraction prompts, optional evaluator prompts, corpus, package/model revisions, source hashes, and checkpoint hashes are preserved. The request helper omits temperature for `gpt-6` names, so a requested extraction temperature must not be described as an effective sampling temperature.

## Main comparison

All values below are final STDI with the existing VAD contribution and unchanged weights. The optional LLM comparator receives the same saved structures **and the full texts**; it can recover information lost in extraction. It is an alternative pipeline, not a pure swap that uses only the extracted relations, and it is not a validated gold standard.

| Scenario and change | Current cluster STDI | Existing LLM-comparator STDI |
| --- | ---: | ---: |
| Government: identical text, reused structure | 0.000000 | 0.000000 |
| Government: equivalent passive-voice paraphrase | 0.215154 | 0.001420 |
| Government: deny the postponement claim | 0.050166 | 0.439332 |
| Government: attribute postponement to the union | 0.221769 | 0.187926 |
| Government: two days → twenty days | 0.121229 | 0.250350 |
| Government: add an intention to weaken the union | 0.240086 | 0.253214 |
| Government: add secret surveillance | 0.231781 | 0.379859 |
| Health: equivalent passive-voice paraphrase | 0.172338 | 0.003036 |
| Health: deny the closure claim | 0.171092 | 0.438610 |
| Election: equivalent passive-voice paraphrase | 0.356553 | 0.000692 |
| Election: deny the removal claim | 0.244031 | 0.438438 |
| Election: reverse the actors' responsibilities | 0.135889 | 0.313077 |
| Election: 120 names → 1200 names | 0.064627 | 0.063041 |

For all three scenarios, the current STDI gives the negation a lower score than the paraphrase. It also gives the quantity change a lower score than the paraphrase in all three scenarios. These are diagnostic order checks for these constructions, not calibrated numerical targets. Of nine checks comparing negation, quantity change, and actor reassignment against paraphrase, the current pipeline matches the intended order in 2/9; the existing LLM evaluator matches it in 9/9. This should not be described as accuracy against human labels.

The optional evaluator assigns zero **structural** drift to all three paraphrases; their small nonzero full STDI values come from VAD. It assigns relation drift 0.75 to the three negations, compared with current relation distances of 0.036764, 0.070466, and 0.070350.

## What happens in the government examples

Original:

> The government postponed negotiations with the teachers union for two days. The union requested a new meeting.

### Equivalent paraphrase

> Negotiations with the teachers union were postponed by the government for two days. A new meeting was requested by the union.

The information is retained in the input text, but the extractor drops the two-day duration and reduces the original's two subtopics to one. It also segments the triples differently:

- Original: government / postponed / negotiations with the teachers union for two days.
- Paraphrase: government / postponed negotiations with / teachers union.
- Original meeting request: teachers union / requested / a new meeting.
- Paraphrase meeting request: teachers union / requested a new meeting / government.

The subtopic distance is 0.649706 and contributes approximately 0.162427 to content drift, explaining most of the full STDI 0.215154. The index is reacting in large part to information lost or reformulated by extraction, not to information removed from the news text.

### Negation

> The government did not postpone negotiations with the teachers union for two days. The union requested a new meeting.

The extractor preserves the negation. Therefore its low current relation drift, 0.036764, cannot be attributed simply to failure to extract “not.” The embedding comparison gives the affirmative and negative relations nearly the same representation for this purpose. Full STDI is only 0.050166.

### Added motive

> The government postponed negotiations with the teachers union for two days. The union requested a new meeting. The government deliberately delayed the talks to weaken the union.

The extractor does retain the unsupported intention as an additional relation:

> government / delayed talks to weaken / teachers union.

Full STDI is 0.240086. Thus the earlier concern that the extractor might categorically discard motives is **not supported by these examples**: all three asserted motives are represented in relations. There are still losses and reframing errors in extraction, but motives are not generally absent here.

The tentative variant receives 0.245879, marginally more than the asserted version. Direct tentative-versus-asserted comparison receives 0.078322. Replacing the motive with protection yields original-relative STDI 0.150556; directly comparing “weaken” with “protect” yields 0.113506, with relation drift only 0.155690 across the complete extracted relation sets.

### New concrete action

> The government postponed negotiations with the teachers union for two days. The union requested a new meeting. The government secretly ordered surveillance of the union.

The extractor records the surveillance, and current full STDI is 0.231781. This example is deliberately allowed in the controlled corpus even though the simulation's rewrite prompt restricts fabrication of new specific events. We are testing measurement sensitivity rather than asking a persona to obey that prompt.

## Same text, different extraction

Three independent extractions were performed for the government original, asserted-motive variant, and negation variant. Every comparison below uses the first original extraction as its reference.

| Input text being extracted | First response | Second response | Third response |
| --- | ---: | ---: | ---: |
| Exactly the original text | 0.000000 | 0.175242 | 0.170049 |
| Original plus asserted motive | 0.240086 | 0.179763 | 0.139023 |
| Negated postponement | 0.050166 | 0.201104 | 0.061276 |

The first original comparison is zero by construction because it reuses the same structure. The two independent original extractions produce nonzero scores despite identical texts and identical VAD. This is extraction-induced variation, not informational change in the message. Three replicates establish a counterexample to invariance here; they are too few to estimate general reliability.

Freezing the original prevents its re-extraction from varying across variants, but does not eliminate variation in extraction of the rewritten text. Both sides of the representation need consistent granularity and adequate preservation of propositions.

## Comparator isolated from extraction

All nine structural probes use a fixed theme, subtopic list, entity set, and no internal contradiction. Only a single manually specified relation changes. VAD is omitted intentionally, and the resulting column is explicitly the STDI **without VAD**, not the complete text pipeline.

| Controlled relation change | Current relation drift | STDI without VAD |
| --- | ---: | ---: |
| Identical relation | 0.000000 | 0.000000 |
| postponed → did not postpone | 0.053929 | 0.013482 |
| government/union roles reversed | 0.238808 | 0.059702 |
| postponement → delay to weaken the union | 0.057832 | 0.014458 |
| postponement → surveillance | 0.303633 | 0.075908 |
| weaken → protect | 0.031233 | 0.007808 |
| may have → deliberately | 0.038281 | 0.009570 |
| two days → twenty days | 0.027636 | 0.006909 |

The remaining probe adds a duration to a relation that originally has none; its relation distance is 0.033703. It is a specificity-addition control, not a two-versus-twenty comparison.

These results demonstrate weak sensitivity in the comparator even when the relevant distinction is provided explicitly in the input structure. In the current code, relation similarity combines whole-triple embeddings (0.45), subject (0.25), action (0.15), and object (0.15). Shared actors and very similar wording can preserve high cosine similarity despite different truth conditions, polarity, quantity, or intent. Clustering assignments do not impose a separate semantic-opposition penalty in `cluster_v2`.

With three of four content components fixed at zero, relation drift contributes only one quarter of its value. Giving relations more weight would amplify the distance, but a negation distance of 0.053929 or opposite-motive distance of 0.031233 would still be small even at weight one. The comparison itself deserves attention before fitting aggregate weights.

## Further extraction and evaluation issues

- Election paraphrase: the original extraction includes `voter register` as an entity and two subtopics; the equivalent paraphrase omits that entity from its entity list and returns no subtopics. Subtopic drift becomes 1.0 and entity drift 0.333333 despite preserved input information.
- Quantity details can move between relations and subtopics or disappear from relations. The government twenty-day extraction drops the duration from the relation but keeps it as a subtopic.
- Government protection-motive extraction invents attribution: a subtopic says “The government said it delayed talks to protect the union,” although the supplied rewrite states the motive directly and never reports a government statement.
- Election negation extraction similarly describes “the commission's denial,” although the input denies the event without identifying the commission as the source of a denial. An assertion made by the message should not silently become an attributed statement.
- All three explicit internal-contradiction variants receive internal-contradiction score 1 from extraction. All three simple negations receive zero. The current extractor correctly separates these two constructs in this corpus.
- Normative recommendations are sometimes extracted as `should` relations. They are additional message content, but should not be confused with an action that occurred. Their intended STDI severity requires a construct definition rather than a universal assumption that all opinions must score zero.
- The existing optional LLM evaluator improves the paraphrase/negation ordering here, but assigns only relation drift 0.25 to the 120→1200 change, leaving full STDI at 0.063041. It is not a comprehensive solution demonstrated by this audit.
- The LLM evaluator uses discrete component levels and can count one change in multiple components. For example, it assigns topic and subtopic drift as well as relation drift to the negations. The fairness of those choices still needs independent review.

## Calibration priorities

1. **Stabilize the extracted representation.** Preserve event propositions, quantities and units, actor roles, polarity, modality, attribution, and added motives consistently. Separate textual claims from externally verified facts. Compare semantically equivalent texts and repeated extractions to quantify the remaining variation.
2. **Evaluate relation matching by meaning, not cosine overlap alone.** Explicitly test affirmative/negative claims, role reversal, opposite motives, changed quantities, and possible-versus-certain claims. Use the existing LLM comparator as one candidate; consider explicit proposition slots or another entailment/contradiction comparison in a separately versioned experiment.
3. **Review subtopic granularity and matching.** The current matching divides by the larger list size. Inconsistent list lengths can produce large scores for equivalent messages. The goal is to preserve distinct information while keeping extraction artifacts from dominating.
4. **Calibrate weights after representation and comparison are evaluated.** The synthetic corpus should become part of a broader held-out evaluation with independent human assessments. Do not choose weights merely to separate persona means or fit these three scenarios.

No threshold or new weight is established as correct. The audit supplies evidence and reproducible cases for the next calibration step while preserving all production behavior.

## Artifacts and reproduction

- `data/synthetic/stdi_information_change_pairs.json`: canonical controlled corpus.
- `scripts/audit_stdi_information_changes.py`: preparation, checkpointed extraction, current scoring, and optional existing-LLM comparison.
- `scripts/report_stdi_information_audit.py`: tables, diagnostic checks, scored examples, and plots.
- `input_pairs.csv` and `pairs_for_review.md`: complete initial pairs.
- `extractions/`: all 42 parsed extraction responses, exact inputs, timestamps, and configuration hashes.
- `scored_pairs.csv`: current scores, component values, both structures, direct contrasts, and repeats.
- `comparator_only_probes.csv`: nine manually specified structural controls.
- `semantic_comparisons/` and `semantic_method_comparison.csv`: all 42 optional comparisons and rationales.
- `change_type_summary.csv`: descriptive summaries over three scenarios per change type.
- `diagnostic_order_checks.csv`: nine predeclared-style pair-order checks, explicitly not human-validated targets.
- `scored_examples.md` and `audit_review.html`: all 36 examples with their extracted structures and scores; HTML includes the interactive comparison plot.
- `run_manifest.json`, `evidence_manifest.json`, and prompt configuration files: provenance and configuration.

From the repository root, after the existing environment is available:

```powershell
uv run --no-sync python scripts/audit_stdi_information_changes.py --stage prepare
uv run --no-sync python scripts/audit_stdi_information_changes.py --stage extract
uv run --no-sync python scripts/audit_stdi_information_changes.py --stage score
uv run --no-sync python scripts/audit_stdi_information_changes.py --stage semantic
uv run --no-sync python scripts/report_stdi_information_audit.py
```

Extraction and semantic stages use the configured API and reuse validated checkpoints. Scoring uses locally cached models. To perform new independent extractions, choose a new `--output-dir`; do not delete or overwrite this audit's evidence. `--export-png` on the report script also exports the plot when the local Plotly image exporter is available.

## Limits

The corpus contains three deliberately simple English scenarios and assistant-authored variants. Each transformation is represented by only three examples, and repeated extraction is examined only for three government texts. There is no independent human gold standard, no model-repetition study for the optional evaluator, and no external factual verification. Findings establish specific failure cases and a useful diagnosis; they do not quantify error rates in real news, determine universal severity thresholds, or establish which replacement method is best.

No thesis or bibliography file was edited, and these local artifacts are not automatically synchronized with Overleaf.
