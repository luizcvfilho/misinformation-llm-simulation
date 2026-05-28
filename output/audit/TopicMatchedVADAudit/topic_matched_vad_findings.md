# Topic-Matched False-to-True VAD Findings

This audit compares VAD scores between original fake news articles and LLM rewrites
that attempt to make the same articles truthful while preserving their topic.

## Sample

- Valid paired articles: 24
- Requested rewrites: 25
- Failed rewrites: 1, due to Gemini free-tier quota exhaustion
- Main topics represented:
  - News: 13 pairs
  - politics: 6 pairs
  - left-news: 3 pairs
  - Government News: 1 pair
  - Middle-east: 1 pair

## Global Results

| Dimension | Fake mean | Rewritten mean | Mean delta | Median delta | Wilcoxon p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| Valence | 2.843 | 2.863 | +0.020 | +0.101 | 0.169 |
| Arousal | 3.292 | 3.096 | -0.196 | -0.138 | 0.000003 |
| Dominance | 3.162 | 3.148 | -0.015 | +0.008 | 0.565 |

## Interpretation

The clearest observed effect is a consistent decrease in arousal after rewriting
fake articles as truthful same-topic articles. Arousal decreased in 21 of the 24
valid pairs, suggesting that the original fake articles tended to use more
emotionally activating, urgent, or inflammatory language.

Valence increased slightly on average, but the effect is small and not
statistically strong in this sample. This suggests that truthful rewrites do not
necessarily become much more positive; instead, they mainly become less
emotionally charged.

Dominance changed very little overall. The rewritten articles retained a similar
level of assertiveness or perceived control, which is plausible because both
versions remain written in a declarative news style.

## Topic-Level Notes

The strongest topic-level signal appears in the `News` subset:

- Mean valence delta: +0.064
- Mean arousal delta: -0.213
- Arousal Wilcoxon p-value: 0.000488

The `politics` subset also shows lower arousal after rewriting, with a mean
arousal delta of -0.175, but the smaller number of pairs makes the evidence less
stable.

## Caveats

This should be treated as a pilot result because the current analysis includes
only 24 valid pairs. The arousal effect is already strong in this small sample,
but the analysis should be rerun with more rows before drawing final conclusions.

The LLM rewrites should also be reviewed for factual quality. A lower-arousal
rewrite is not automatically a fully truthful article; it may simply be more
neutral or less sensational.

## Preliminary Conclusion

For same-topic fake-to-truthful rewrites, the main VAD shift is not that articles
become substantially more positive. The main shift is that they become less
emotionally activating. This supports the hypothesis that fake news in this
sample relies heavily on high-arousal framing, and that correcting the text
reduces that affective intensity.
