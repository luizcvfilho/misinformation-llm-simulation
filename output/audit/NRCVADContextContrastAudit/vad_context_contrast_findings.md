# NRC VAD v2.1 Context Contrast Audit

## Method and sample

We evaluated 40 synthetic English texts, organized into 20 pairs with the same focal event and positive/negative contexts. There are 40 texts with available scores. These are exploratory results of this study.

The official downloaded file contains 54801 entries, including 10073 multiword expressions. The manifest records hashes of the lexicon, input, and BERT reference.

The scorer averages recognized term occurrences, selecting the longest available expression at each position without counting its words again. Repeated terms count again. Normalization uses NFKC, case folding, and equivalent apostrophes; punctuation interrupts expressions. No lemmatization, negation handling, stopword removal, or truncation is applied. Unknown terms are excluded; texts without matches receive missing scores.

Native values from -1 to 1 are stored in `nrc_native_*`. To compare scale amplitudes, `vad_*` columns use `3 + 2 * native_score`, on the 1 to 5 scale. This transformation does not calibrate the methods or establish equivalence between their scores.

## Lexical coverage

Token-weighted overall coverage was 69.3%; minimum per-document coverage was 55.6%. We recognized 26 multiword expression occurrences. Coverage measures lexical recognition, not affective assessment quality.

## Differences between contexts

Delta = positive context - negative context. The means below use the 1 to 5 scale.

| Dimension | Positive mean | Negative mean | Mean delta | Mean absolute delta | Pairs with delta > 0 | Wilcoxon p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| valence | 3.271 | 2.979 | +0.291 | 0.291 | 20/20 | 0.000002 |
| arousal | 2.946 | 2.863 | +0.082 | 0.119 | 15/20 | 0.013617 |
| dominance | 3.077 | 2.882 | +0.195 | 0.210 | 17/20 | 0.000036 |

Valence was higher in the positive context in 20 of 20 valid pairs. Its mean delta was +0.291. This contrast is consistent with the example design, but still requires validation on independent texts and against human assessments.


## Comparison with the saved BERT audit

Texts and identifiers were checked against the historical CSV. BERT was not rerun. The ratio below compares mean absolute pair differences, divided by the same amplitude of 4; it does not measure accuracy.

| Dimension | NRC mean delta | BERT mean delta | NRC normalized absolute difference | BERT normalized absolute difference | NRC/BERT ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| valence | +0.291 | +0.410 | 0.0728 | 0.1025 | 0.71 |
| arousal | +0.082 | -0.052 | 0.0299 | 0.0250 | 1.20 |
| dominance | +0.195 | +0.141 | 0.0525 | 0.0360 | 1.46 |

Comparison of mean magnitudes and directions:

- **valence**: mean absolute difference is smaller with NRC; mean direction is the same relative to BERT.
- **arousal**: mean absolute difference is larger with NRC; mean direction is opposite relative to BERT.
- **dominance**: mean absolute difference is larger with NRC; mean direction is the same relative to BERT.

Direction disagreements should be inspected in the texts. This dataset provides no human reference for deciding which method is correct for arousal or dominance.


## Examples with the largest differences

| Focal event | Dimension with the largest change | Absolute difference |
| --- | --- | ---: |
| A building showed cracks | valence | 0.867 |
| A corn harvest had partial losses | valence | 0.753 |
| A factory laid off 30 workers | valence | 0.457 |
| A storm flooded two streets | arousal | 0.421 |
| A player was injured during training | valence | 0.416 |

## Interpretation and limitations

A larger difference indicates greater sensitivity on this dataset, but does not demonstrate greater validity. The lexicon measures affective vocabulary associations and may confound news subject matter with framing. Direction, magnitude, and human agreement must be assessed separately for each dimension. Paired tests are exploratory, two-sided, and uncorrected for multiple comparisons; the contexts do not guarantee opposite directions for arousal and dominance.

The evaluation uses English only and does not validate Portuguese. No matches do not imply neutrality. The mean may hide coexisting terms with opposing associations. These results do not verify factual veracity, belief, exposure, or sharing. The audit does not change the default scorer or previous STDI results.

## References

- Mohammad (2025), [NRC VAD Lexicon v2](https://arxiv.org/abs/2503.23547).
- Mohammad (2025), [Breaking Bad: Norms for Valence, Arousal, and Dominance for over 10k English Multiword Expressions](https://aclanthology.org/2025.ijcnlp-long.107/).
- [Official resource page and terms of use](https://saifmohammad.com/WebPages/nrc-vad.html). The lexicon stays in the local Git-ignored cache; output files do not redistribute its entry table.

