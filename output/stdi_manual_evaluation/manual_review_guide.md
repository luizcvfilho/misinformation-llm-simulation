# STDI Manual Review

Each row compares `original_text` with `modified_text`. The `target_metric` identifies the type of controlled change requested from the rewrite model.

Fill these columns in `scored_stdi_pairs.csv` after reviewing the pair:

- `manual_target_metric_score`: intensity from 0 to 1 of the requested metric change.
- `manual_expected_stdi`: overall semantic drift from 0 to 1, based on your judgement.
- `manual_review_status`: set to `reviewed` when the annotation is complete.
- `manual_notes`: explain ambiguity, failed rewrites, or unexpected side effects.

Use the same rubric across all 50 rows. The regression uses only non-empty `manual_expected_stdi` values and the STDI component values calculated by the pipeline.
