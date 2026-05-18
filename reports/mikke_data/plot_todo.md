# Plot Todo

This file tracks report plots, their data readiness, and whether a figure is final or only useful for design. The earlier broad candidate list has been removed so the first planning target is the main ablation study.

Status labels:

- `ready_for_draft`: Existing data appears usable for draft plotting.
- `placeholder`: Useful for plot design, but not final report evidence.
- `incomplete`: Some data exists, but more runs or aggregation are needed.
- `not_started`: No curated data has been prepared yet.
- `final_ready`: Curated data and figure are ready for final report use.

## Main Plot: Ablation Study

Figure id: `ablation_kshot_strict_f1`

Goal: show how each model changes CoNLL2003 strict span F1 across k-shot settings. Each line should be the mean over the three split seeds for that model and k value.

Recommended first draft format:

- Use a line plot with x-axis values `k=5`, `k=10`, `k=20`, and `k=50`.
- Use a continuous y-axis from `0` to `100`.
- Label the y-axis `Strict span F1`.
- Plot one line per model, where each point is the mean over split seeds `42`, `142`, and `242`.
- Use solid lines for draft-ready test data and dashed/open-marker lines for placeholder validation data.
- Use the method colors from `styleguide.md`.

Draft method order:

1. `BERT`
2. `Pat+Perm BART (VAL)`
3. `Pat+Perm BART ensemble vote`
4. `Pat+Perm BART ensemble → BERT`
5. `Verbalized ensemble majority`
6. `Verbalized BART ens → BERT`
7. `soft_labels BART ens → BERT`

## Current Data Readiness

`BERT`

- Status: `ready_for_draft`
- Source: `results/bert_conll2003_standard_report/BERT_kshot_standard_report.csv`
- Coverage: k5/k10/k20/k50, three split seeds, validation and test.
- Note: Use test strict span F1 and average over the three split seeds per k.

`Pat+Perm BART ensemble vote`

- Status: `ready_for_draft` for `k5_seed42` only.
- Source: `reports/ensemble_tagging/k5_seed42_oada_style_seed_ensemble_verbalized_final2000_test_metrics.json`
- Available values: k5 test strict span F1 for split seeds 42, 142, and 242.
- Note: Excluded from the k-shot line plot until it has all k values and three split seeds.

`Verbalized ensemble majority`

- Status: `ready_for_draft` for `k5_seed42` only.
- Source: `reports/ensemble_tagging/k5_seed42_pet_oada_verbalized_final2000_test_metrics.json`
- Available value: test strict span F1 `0.4841`.
- Note: Excluded from the k-shot line plot until it has all k values and three split seeds.

`Pat+Perm BART ensemble → BERT`

- Status: `ready_for_draft`
- Source: `reports/mikke_data/data/main_ablation_mean_scores.csv`
- Available values: test strict span F1 for k5/k10/k20/k50 and split seeds 42/142/242 from `top_1000 hard_argmax` distillation summaries.
- Note: Included as a full line in the k-shot plot.

`Pat+Perm BART (VAL)`

- Status: `placeholder`
- Source: `reports/mikke_data/data/main_ablation_mean_scores.csv`
- Coverage: validation pilot values exist for k5/k10/k20/k50 and split seeds 42/142/242.
- Note: Included as a dashed placeholder line because test decoding is not available in the summary.

`Verbalized BART ens → BERT`

- Status: `incomplete`
- Source: `reports/bert_distilled_k5_variants.log`
- Note: Log contains k5 distilled runs and validation summaries, but no curated final CSV exists yet.

`soft_labels BART ens → BERT`

- Status: `not_started`
- Source: `docs/learnings/soft_labels.md`
- Note: The soft-label direction was shelved; keep it as an ablation concept only if the report explains why it was not pursued.

## Next Data Work

1. Compile a draft `curated/ablation_k5_results.csv` from the ready and placeholder sources.
2. Generate `plots/ablation_kshot_strict_f1_data.csv` from the curated CSV.
3. Render a draft line plot from the plot-ready CSV.
4. Review labels and colors before adding incomplete models.

## Plot Data Rules

Every final plot should have:

1. A plot-ready CSV in `plots/` with the exact data used for the figure.
2. A source curated CSV in `curated/`.
3. A script in `scripts/` that regenerates the plot from the curated or plot-ready CSV.
4. Explicit status fields showing whether rows are final, placeholder, or incomplete.
