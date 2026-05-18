# Agreement-Filtered Distillation

## Finding

Using the full unlabeled pool for BERT distillation is not automatically best. For `k5_seed42`, high-agreement filtering of the verbalized `final2000` BART ensemble teachers substantially improved the distilled BERT student.

Teacher source:

- BART-base PET/OADA verbalized ensemble.
- `final2000` training setting.
- `k5_seed42`.
- 10 ensemble members.
- Hard-argmax teacher labels aggregated from member predictions.

Subset policy:

- Rank examples by ensemble vote agreement.
- Prefer examples with at least one predicted entity.
- Keep up to 10% high-confidence no-entity rows.
- Preserve original row order after selection.

## Results

Distilled BERT test strict F1:

- Full unlabeled pool, 14,031 rows: `0.5702`
- `top_500`: `0.6015`
- `top_1000`: `0.6640`
- `top_2000`: `0.6257`
- `top_5000`: `0.5905`

Validation strict F1:

- Full unlabeled pool: `0.5506`
- `top_500`: `0.5702`
- `top_1000`: `0.6665`
- `top_2000`: `0.6330`
- `top_5000`: `0.5776`

The best setting is `top_1000`, which is `1000 / 14031 = 7.1%` of the available unlabeled pool. This should be reported both as an absolute count and as a pool fraction so the method can transfer more cleanly to other datasets.

## Report Protocol

Use validation results to justify subset-size selection and plots. The scientifically clean protocol is:

1. Use validation to compare subset sizes and lock `top_1000`.
2. Do not keep tuning subset size based on test scores.
3. After the setting is locked, report test F1 for the final chosen setting across splits/seeds.

It is acceptable to include the k5 pilot test numbers as exploratory evidence if clearly labeled, but the main report claim should rely on the locked protocol: validation for choosing `top_1000`, test for final evaluation.

## Plot Idea

A useful report plot:

- x-axis: teacher subset size (`500`, `1000`, `2000`, `5000`, `full`)
- y-axis: distilled BERT span strict F1
- separate lines or markers for validation and test
- annotate `top_1000` as `7.1%` of the unlabeled pool

Expected visual story: quality peaks at a curated high-agreement subset, then degrades as noisier pseudo-labels are added.

## Next Use

Move forward with `top_1000 hard_argmax` as the default filtered-teacher setting for verbalized `final2000` distillation. Revisit the absolute size only if larger datasets make `1000` too small; in that case compare against an equivalent pool fraction near `7%`.

