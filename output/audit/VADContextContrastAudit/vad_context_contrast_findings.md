# Synthetic Context Contrast VAD Findings

This experiment tested whether VAD scores change when the same focal news event is placed in positive or negative surrounding context.

The first version used Portuguese synthetic news, but the scores stayed very close to `3.0` across all VAD dimensions. Since the current model, `RobroKools/vad-bert`, is based on an English BERT model, this likely compressed the predictions toward neutral values. After rewriting the same 20 paired examples in English, the model became much more sensitive to the contextual framing.

## Main Result

The English synthetic dataset produced clearer differences between positive and negative contexts:

| Context | Documents | Valence mean | Arousal mean | Dominance mean |
| --- | ---: | ---: | ---: | ---: |
| Negative | 20 | 2.460 | 3.161 | 2.800 |
| Positive | 20 | 2.870 | 3.109 | 2.940 |

The strongest change appeared in **valence**. Positive contexts increased average valence by about `0.410` points compared with negative contexts. This suggests that the model does capture whether the broader framing makes the same event feel more positive or more negative.

Negative contexts also produced slightly higher **arousal** and lower **dominance**. This is consistent with the idea that crisis-oriented framing tends to feel more intense and less controlled.

## Interpretation

The VAD dimensions should be read approximately as a `1` to `5` scale, where `3` is neutral:

- **Valence**: emotional pleasantness, from negative to positive.
- **Arousal**: emotional intensity, from calm to activated.
- **Dominance**: perceived control, from powerless to in control.

In this experiment, the same focal event often received different VAD scores depending on the surrounding context. For example, the pair **"A dog died"** had one of the largest shifts:

| Context | Valence | Arousal | Dominance |
| --- | ---: | ---: | ---: |
| Positive | 2.696 | 3.102 | 2.890 |
| Negative | 2.017 | 3.258 | 2.620 |

The focal event remained negative in both cases, but the negative context made it substantially more negative, more emotionally activated, and less controlled.

## Conclusion

The results suggest that VAD can detect contextual emotional framing when the input language matches the model language. The Portuguese run should not be treated as evidence that VAD is insensitive to context; it is more likely evidence that this specific VAD model is not appropriate for Portuguese text without translation or a multilingual/Portuguese VAD model.

For future experiments, the safest setup is to either:

- run this VAD model on English texts or carefully translated texts;
- or replace it with a model validated for Portuguese.
