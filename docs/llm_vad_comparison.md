# Current BERT versus LLM VAD pilot

This exploratory workflow keeps the existing VAD scorer and STDI unchanged.
It compares complete saved texts scored by `RobroKools/vad-bert` and an LLM,
initially `gpt-6-luna` through the project's `chatgpt` provider.

## Scope and outbound payload

- 20 synthetic context pairs from `data/synthetic/vad_context_contrast_news.csv`.
- 30 original/final pairs: the same five jointly available news items across
  SSSS, CCCC, PPPP, DDDD, CCPP, and PPCC in the saved simulation comparison.
- 50 pairs correspond to 75 unique complete English texts after deduplication.
- Sampling is fixed before scoring by SHA-256 ordering of shared news IDs.
- Each external API request sends one complete text and the frozen VAD rubric,
  without persona, title, paired text, expected polarity, or BERT scores.
- The `chatgpt` provider uses `https://api.openai.com/v1`. API credentials are
  used by the SDK and are not included in reports or request text.
- `input_pairs.csv` and `evaluation_jobs.json` contain the exact reviewable text
  payload. There is one successful assessment per unique text, with up to four
  attempts for API/validation failures. Retries are not independent replicates.

## Rubric and comparison

The rubric version is `llm_vad_expressed_tone_v1`. It measures the affective tone
expressed by the narrator/speaker, including implicit appraisal grounded in
wording, rather than observed reader reactions or factual truth. V, A, and D
have nominal ranges of 1–5. Neutral valence does not imply midpoint arousal.
Dominance concerns conveyed emotional agency/control versus helplessness;
it is not certainty, credibility, institutional authority, or an actor's rank.
The BERT model's annotation perspective is not assumed to match this rubric.

Each dimension requires a brief rationale and may include exact contiguous
quotes from the input. Scores outside [1,5], booleans, numeric strings, invalid
JSON/schema, and invented evidence are rejected. A genuinely unrateable
dimension can be null; lack of a directional cue is not automatically missing.

For both estimators, the signed difference is compared minus reference. The
normalized drift is the mean of `min(abs(delta)/4, 1)` over the three dimensions,
matching the current theoretical amplitude normalization. Context pairs use
positive minus negative; simulation pairs use step 4 minus original. Missing
dimensions remain missing, including the aggregate VAD drift. Summaries report
metric-specific joint denominators without neutral or zero imputation.

## Files and execution

```bash
uv run python scripts/compare_llm_vad.py --stage prepare
uv run python scripts/compare_llm_vad.py --stage all
uv run python scripts/compare_llm_vad.py --stage report
```

Use `--scope context` for the 40 synthetic texts alone, `--scope simulation` for
saved simulation pairs, or the default `--scope both`. `--news-count` adjusts
the shared-news sample. Changed inputs/configuration require a new
`--output-dir`; model and provider are configurable. `prepare` makes no API calls.
`all` evaluates/resumes; `report` regenerates reports from saved responses,
reusing the current-revision BERT cache. BERT computation is local and retains
the current 512-WordPiece truncation, recorded per text. Valid LLM responses
are reused only for identical configuration and exact input text.

Default output: `output/audit/LLMVADComparison_20261006/`.

| File | Purpose |
| --- | --- |
| `configuration.json` | Frozen rubric, model/provider, input/source hashes, sampling policy |
| `input_pairs.csv` | Exact paired texts, IDs, dataset, chain, and news membership |
| `evaluation_jobs.json` | Deduplicated full-text payloads in blinded hash order |
| `llm_responses/*.json` | Raw responses, timestamps, attempts, evidence, rationales, failures |
| `text_scores.csv` | Individual BERT/LLM VAD, missing values, and truncation flags |
| `pair_comparison.csv` | Signed deltas, normalized drifts, and original input texts |
| `pair_summary.csv` | Joint descriptive results by dataset and simulation chain |
| `score_agreement.csv` | Descriptive estimator agreement per dimension |
| `manifest.json` | Completion counts, BERT revision/runtime, selected news, limitations |
| `findings.md` | Current descriptive findings and interpretation limits |
| `comparison_dashboard.html` | Self-contained offline plots and inspectable disagreement cases |

`notebooks/llm_vad_comparison_workbench.ipynb` loads these artifacts and provides
paired-drift and individual-score plots plus inspection of disagreement cases.
It does not make external API calls unless `RUN_EVALUATION` is set to true.
The CLI also exports `comparison_dashboard.html`, with Plotly embedded for
offline viewing. Regenerating the report/dashboard makes no LLM calls.

## Interpretation limits

BERT is a comparator, not a human reference. Larger drift or correlation with
BERT does not establish accuracy. Equal scale amplitudes do not calibrate
estimators. Full LLM input and truncated BERT input may differ for long texts.
The five simulation news items are the independent news units; chain pairs
share original texts. The synthetic polarity manipulation supplies a valence
direction control, not a guaranteed arousal/dominance direction. This first
pilot has no independent human ratings, no successful repeated assessments,
and no Portuguese validation. It does not measure actual reader impact,
sharing, belief, or factual veracity.

Inspect `manifest.json` before interpreting tables. Pending/failed evaluations
remain visible, and a prepared report with missing LLM scores is not a result.

## Executed pilot on 2026-10-06

All 75 unique texts received a valid assessment from `gpt-6-luna`; all 50 pairs
have complete VAD for both estimators. These are results of this study.

| Subset | Joint pairs | Current BERT mean drift | LLM mean drift |
| --- | ---: | ---: | ---: |
| Synthetic context controls | 20 | 0.05450 | 0.06250 |
| Saved simulation final versus original | 30 | 0.01986 | 0.07167 |
| Simulations excluding BERT-truncated pairs | 25 | 0.02049 | 0.07000 |

Five rewritten texts, all from the pollster/presidency news item, exceed BERT's
512-WordPiece limit. The original texts in this sample were not truncated.
Excluding the five affected pairs leaves a similar descriptive estimator gap;
it does not isolate all model/rubric differences or balance chain counts.

Positive-minus-negative valence was positive in 20/20 controls for BERT and
19/20 for LLM, with one LLM tie and no reversal. Mean signed valence changes
were +0.41011 and +0.47500 respectively on the 1–5 scales. Across the 75 texts,
mean arousal was 3.09036 for BERT and 2.24267 for LLM, indicating a substantial
level difference that is distinct from paired drift. Higher simulation drift
does not establish greater validity or justify replacing the production scorer.

For detailed qualitative caveats, see
`output/audit/LLMVADComparison_20261006/interpretation.md`. In particular,
some dominance rationales warrant review for actor-agency versus narrator-agency
mixing despite the fixed rubric. This pilot used one successful rating per
text; repeated ratings and independent human judgments remain future validation.
