# VAD Comparison on Saved Simulation Texts

We analyzed 6 executions, 50 distinct news items, and 1200 saved steps. Chains: CCCC, CCPP, DDDD, PPCC, PPPP, SSSS. These are exploratory results of this study.

Texts were reused in full. BERT was not rerun, and no rewriting or structural extraction was regenerated. BERT identity comes from historical documentation when it is not recorded in the files.

BERT scores were normalized using `(score - 1)/4`; NRC scores using `(native_score + 1)/2`. Both scales range from 0 to 1. Drift is the mean absolute difference across the three dimensions. The transformation uses theoretical amplitudes and does not apply observed min/max normalization. It does not establish equivalence or calibration between the estimators.

Each step is compared with the original news item and its actual input text. Hypothetical STDI replaces only the VAD contribution, preserving the recorded non-affective contribution and the weight of 0.20. The historical formula and BERT drift recalculated from saved scores are checked before this replacement. Missing values and failures are not filled with zero.

## Last recorded step per news item and execution

| Chain | Valid pairs | BERT VAD | NRC VAD | BERT STDI | Hypothetical NRC STDI |
| --- | ---: | ---: | ---: | ---: | ---: |
| CCCC | 50 | 0.0172 | 0.0187 | 0.2564 | 0.2566 |
| CCPP | 50 | 0.0190 | 0.0180 | 0.2500 | 0.2499 |
| DDDD | 50 | 0.0169 | 0.0178 | 0.2394 | 0.2395 |
| PPCC | 50 | 0.0181 | 0.0183 | 0.2580 | 0.2580 |
| PPPP | 50 | 0.0196 | 0.0164 | 0.2687 | 0.2683 |
| SSSS | 50 | 0.0246 | 0.0183 | 0.2530 | 0.2520 |

There are 1200 steps with comparable STDI. Across these steps, mean STDI change was -0.00018; maximum absolute change was 0.00846.

## Limitations and inspection

More variation does not demonstrate greater validity. NRC aggregates lexical associations, whereas BERT uses context. NRC processes full texts, while historical BERT may have truncated at 512 tokens; this difference was not isolated. The same news items appear across chains and steps, so steps are not independent observations. This comparison has no human validation.

Per-text coverage and signed differences are in `step_comparison.csv`; `chain_step_summary.csv` aggregates only observations available for both methods. Formula failures block only hypothetical STDI. Cumulative STDI sums incremental changes; it can exceed 1 and is not equivalent to distance from the original.

The evaluation is restricted to English, does not validate Portuguese, and does not measure factual veracity, belief, exposure, or sharing. Original scores remain in the simulation files. The notebook includes charts and example selection.

