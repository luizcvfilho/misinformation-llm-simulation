# Interpretation of the completed full-text VAD pilot

These are exploratory results of this study, not external literature findings.
75/75 unique English texts received valid `gpt-6-luna` VAD assessments; the 50
pairs have complete V,A,D for both estimators. The LLM rubric was frozen before
scoring and no production STDI or VAD estimator was changed.

| Subset | Joint pairs | BERT mean normalized drift | LLM mean normalized drift |
| --- | ---: | ---: | ---: |
| Synthetic context controls | 20 | 0.05450 | 0.06250 |
| Saved simulation final versus original | 30 | 0.01986 | 0.07167 |
| Simulations without BERT-truncated inputs | 25 | 0.02049 | 0.07000 |

## What the differences support

In this sample the LLM produced larger paired changes on the saved simulations,
approximately 3.61 times the BERT aggregate drift. The gap remained after
excluding the five pairs whose rewritten texts exceeded BERT's input limit.
The originals were not truncated. All five truncated rewrites belong to the
same pollster/presidency news item; exclusion leaves unequal chain counts.
These descriptive comparisons do not establish an isolated model effect.

The context controls show a smaller estimator difference. Positive-minus-negative
valence increased in 20/20 BERT controls and 19/20 LLM controls. The LLM tied on
`context_5` (robbery amid contrasting public-safety contexts); it did not reverse
the valence direction. Mean signed changes were +0.41011 for BERT and +0.47500
for LLM on the 1–5 scales. This supports directional responsiveness to this
synthetic manipulation, not accuracy on natural news.

Across all 75 unique texts, mean arousal was 3.09036 for BERT and 2.24267 for
LLM. This level offset differs from drift and illustrates why matching nominal
scale bounds does not calibrate annotators. Rank agreement was approximately
0.675 for valence, 0.413 for arousal, and 0.415 for dominance. Agreement with
BERT is not human-grounded accuracy.

## Inspectable examples and caution about dominance

For `simulation_DDDD_80a390c3c68815dadd42b1ea065fd1af` (sidewalk repairs),
BERT drift is 0.01917 while LLM drift is 0.23333. LLM scores change from
V=3.6,A=1.8,D=3.7 to V=2.4,A=2.7,D=3.0. The rewritten passage introduces
a skeptical appraisal about institutional control over funds and the accepted
narrative. The LLM's valence and arousal rationales explicitly discuss those
wording changes. This is an inspectable disagreement case, not a verified error
of either evaluator.

The original dominance rationale refers to the city's capacity to address
repairs, whereas the rewritten dominance rationale separates institutional
control from personal agency. This may mix actor-agency with narrator-agency,
despite the rubric requesting the narrator's conveyed emotional control.
Human review is needed before interpreting such changes as a valid dominance
measure. Valid JSON, in-range numbers, and grounded quotes do not validate
the psychological construct.

## Implications for the next experiment

The first comparison demonstrates that the LLM scorer can yield a different
VAD profile without changing the drift formula. It does not justify multiplying
weights or adopting the scorer simply because its drift is larger. Retain the
current production estimator while reviewing the rubric and assessing repeated
LLM ratings against independently annotated news pairs. In particular, keep
emotion expressed in text distinct from actual reader impact, factual truth,
belief, exposure, and sharing behavior. This sample contains five independent
simulation news items and one successful LLM rating per unique text.

`findings.md` and the CSV tables are generated from raw checkpointed responses.
`comparison_dashboard.html` includes offline plots, the frozen rubric, and
expandable disagreement cases. This interpretation file records qualitative
review separately from the automatic numerical report.
