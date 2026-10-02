# Three-model context contrast comparison

20 synthetic English pairs; 20 jointly valid pairs. The current BERT revision is `c4c659fd17dd5807572b3f0f577de8c408401ca7`.

BERT: (score-1)/4; NRC v2.1: (native+1)/2; MEmoLon: (native-1)/8. Common nominal 0-1 amplitude; no sample min/max scaling or calibration.

Positive minus negative context; raw deltas below use the common 1-5 scale.

| Model | Dimension | Mean signed delta | Mean absolute delta | Normalized absolute delta | Positive delta pairs |
| --- | --- | ---: | ---: | ---: | ---: |
| Current BERT | valence | +0.4101 | 0.4101 | 0.1025 | 20/20 |
| Current BERT | arousal | -0.0522 | 0.0998 | 0.0250 | 6/20 |
| Current BERT | dominance | +0.1405 | 0.1440 | 0.0360 | 19/20 |
| NRC v2.1 | valence | +0.2913 | 0.2913 | 0.0728 | 20/20 |
| NRC v2.1 | arousal | +0.0824 | 0.1194 | 0.0299 | 15/20 |
| NRC v2.1 | dominance | +0.1952 | 0.2101 | 0.0525 | 17/20 |
| MEmoLon MTL_grouped | valence | +0.0972 | 0.0982 | 0.0246 | 19/20 |
| MEmoLon MTL_grouped | arousal | -0.0089 | 0.0240 | 0.0060 | 7/20 |
| MEmoLon MTL_grouped | dominance | +0.0728 | 0.0728 | 0.0182 | 20/20 |

NRC and MEmoLon token coverage: 69.31% and 100.00%. Coverage does not establish validity. BERT is a comparator, not a human gold standard. Direction and magnitude must be examined separately. Paired statistical tests are exploratory and uncorrected for multiple comparisons. Portuguese is not validated.

Individual scores and missing pairs for all three models are retained in the joint tables; only unified results are exported.
