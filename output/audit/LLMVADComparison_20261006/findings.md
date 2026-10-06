# Current BERT versus LLM VAD: exploratory full-text pilot

LLM: `gpt-6-luna` via `chatgpt`; rubric `llm_vad_expressed_tone_v1`. Requested: one successful assessment per unique text.

Execution status: **complete**. Pending/failed evaluations are not comparison results.

50 pairs and 75 unique texts. LLM statuses: {'valid': 75}.

Context pairs compare positive minus negative context. Simulation pairs compare step 4 against the original, using the same shared news across six chains. The hash-based sample was selected before scoring.

Both use complete saved texts, scored independently, with nominal 1-5 scales. BERT retains its existing 512-token truncation; LLM uses full text. 5 unique texts exceed BERT's limit. No segmentation, new rewrites, or STDI changes.

| Dataset | Chain | Joint pairs | BERT mean VAD drift | LLM mean VAD drift |
| --- | --- | ---: | ---: | ---: |
| context | all | 20 | 0.05450 | 0.06250 |
| simulation | all | 30 | 0.01986 | 0.07167 |
| simulation | all_untruncated | 25 | 0.02049 | 0.07000 |
| simulation | CCCC | 5 | 0.01389 | 0.06667 |
| simulation | CCPP | 5 | 0.02165 | 0.04500 |
| simulation | DDDD | 5 | 0.02334 | 0.11000 |
| simulation | PPCC | 5 | 0.01620 | 0.05000 |
| simulation | PPPP | 5 | 0.01788 | 0.07500 |
| simulation | SSSS | 5 | 0.02622 | 0.08333 |

## Dimension changes

Signed deltas use native 1-5 score units; dimension drifts use normalized 0-1 units. Joint pairs are counted separately for each metric.

| Dataset | Metric | Joint pairs | BERT mean | LLM mean |
| --- | --- | ---: | ---: | ---: |
| context | valence_delta | 20 | 0.41011 | 0.47500 |
| context | arousal_delta | 20 | -0.05219 | -0.12500 |
| context | dominance_delta | 20 | 0.14055 | 0.03000 |
| context | valence_drift | 20 | 0.10253 | 0.11875 |
| context | arousal_drift | 20 | 0.02495 | 0.06125 |
| context | dominance_drift | 20 | 0.03601 | 0.00750 |
| simulation | valence_delta | 30 | -0.04200 | -0.11667 |
| simulation | arousal_delta | 30 | -0.02688 | 0.11000 |
| simulation | dominance_delta | 30 | 0.01679 | 0.09333 |
| simulation | valence_drift | 30 | 0.02411 | 0.09917 |
| simulation | arousal_drift | 30 | 0.02237 | 0.05917 |
| simulation | dominance_drift | 30 | 0.01311 | 0.05667 |

Context-control positive valence deltas: BERT 20/20; LLM 19/20.


## Interpretation limits

5 pairs involve at least one BERT-truncated input. These pair-level gaps mix estimator and input-length effects.

The `all_untruncated` sensitivity row excludes those pairs, without rescoring or changing the rubric. Exclusion can make chain counts unequal; it is descriptive.

Larger drift is not evidence of better accuracy. BERT is a comparator, not a human gold standard. Equal nominal scales do not calibrate the estimators or guarantee identical annotation perspectives. Scores estimate expressed affective tone, not observed reader response, factual veracity, or sharing behavior. Neutral valence need not imply midpoint arousal. Dominance refers to the narrator's conveyed agency, not institutional power or certainty.

There are no independent human ratings or repeated successful assessments in this pilot. Validation retries are not independent replicates. Simulation pairs share original news; pair counts are not independent sample sizes. Arousal and dominance have no assumed direction in the positive/negative context controls. Agreement correlations are descriptive, not accuracy measures. Portuguese is not validated. Missing dimensions remain missing.

## Reproduction and inspection

`configuration.json` contains the frozen rubric and source hashes; `input_pairs.csv` contains exact texts; `llm_responses/` preserves raw responses, validation attempts, rationales, evidence, and missing values. `text_scores.csv`, `pair_comparison.csv`, `pair_summary.csv`, and `score_agreement.csv` retain individual and joint results. Resuming reuses valid responses for identical configuration and texts.
