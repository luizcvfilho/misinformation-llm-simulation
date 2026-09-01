# STDI logistic-regression analysis

The classes are reference groups for analyzing STDI features, not a factual verifier.
ROC-AUC: 0.921
Average precision: 0.925

Top variables by permutation importance:
- vad_arousal: 0.3135
- word_count: 0.1094
- has_internal_contradiction: 0.0070
- entity_count: 0.0051
- relation_count: 0.0019

TF-IDF ablation (same holdout split):
- stdi_only: ROC-AUC 0.921
- tfidf_only: ROC-AUC 1.000
- stdi_plus_tfidf: ROC-AUC 0.980
