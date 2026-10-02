# Three-model VAD comparison on saved simulations

6 executions, 50 news items, 1200 steps, and 1204 unique English texts.

BERT: (score-1)/4; NRC v2.1: (native+1)/2; MEmoLon: (native-1)/8. Common nominal 0-1 amplitude; no sample min/max scaling or calibration.

All estimators use the same saved texts and current BERT revision. Only VAD is replaced in hypothetical STDI; historical formula checks and recorded non-VAD contributions are preserved. No LLM calls.

| Model | Joint steps | Mean original-relative VAD drift | Mean hypothetical STDI | Rewritten token coverage |
| --- | ---: | ---: | ---: | ---: |
| Current BERT | 1200 | 0.01740 | 0.23569 | N/A |
| NRC v2.1 | 1200 | 0.01612 | 0.23551 | 63.59% |
| MEmoLon MTL_grouped | 1200 | 0.00460 | 0.23376 | 98.25% |

BERT truncates at 512 WordPiece tokens; lexical estimators use full texts. 39 unique inputs exceeded this limit. Repeated news and steps are dependent. Greater variation or proximity to BERT does not establish accuracy. There are no independent human VAD ratings. These results do not validate Portuguese or factual veracity.

Individual scores and missing values for all three models are retained in the joint step table; only unified results are exported.
