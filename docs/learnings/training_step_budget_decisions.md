# Training Step-Budget Decisions

## BART Teacher Budget

The BART teacher budget was locked at `2000` steps after the early OADA-XE/XE pilot runs. The original interactive canvas no longer contains a rendered plot artifact, but the canvas source preserved the mean mini-val fitness curves used for the decision:

```text
C:\Users\meatl\.cursor\projects\c-Users-meatl-Documents-ITU-nlp-fire-horses-nlp-burninghorses\canvases\oada-evidence-report.canvas.tsx
```

Those curves have been converted into reproducible report artifacts:

- `reports/mikke_data/data/bart_step_budget_fitness_curves.csv`
- `reports/mikke_data/plots/bart_step_budget_fitness_curves.svg`
- `reports/mikke_data/scripts/plot_bart_step_budget_decision.py`

The plotted data are mean mini-val strict span F1 curves across three seeds for fixed-2000 `pattern_01` BART pilots:

- `XE fixed-2000`
- `OADA-XE cap24 fixed-2000`

The evidence supports `2000` as a conservative fixed budget:

- Many runs continue improving past the early pilot horizon.
- Several individual runs peak before 2000, so `2000` is not universally optimal.
- The mean curves are stable enough near 2000 to justify a fixed budget.
- A fixed final-step policy is cleaner for report-grade teacher generation than selecting per-run checkpoints from mini-val.

This decision is also summarized in `reports/oada_xe_pilot_findings.md`:

- OADA-XE best mini-val steps by k: `1800/1500/1700`, `1700/1200/500`, `600/1400/1600`, `1200/1800/1000`.
- XE best mini-val steps by k: `1300/1000/1200`, `300/1500/400`, `1700/1400/1600`, `1100/1900/1000`.

## BERT Student Budget

BERT was later handled separately from BART. The final report-grade BERT policy is `600` steps with final-checkpoint evaluation, no early stopping, and no best-checkpoint selection.

The rationale is documented by:

- `reports/mikke_data/data/bert_baseline_stopping_analysis.csv`
- `scripts/update_bert_baseline_fixed600_ablation.py`

The stopping analysis showed that early stopping often under-trained the gold-only BERT baseline. For the report, using a fixed `600`-step final checkpoint made the BERT baseline and distilled BERT students more comparable.

## Reporting Wording

Recommended report phrasing:

> BART teacher models were trained with a fixed 2000-step budget, chosen from pilot mini-validation fitness curves showing continued improvement or stable performance near the end of the run. BERT students and baselines used a separate fixed 600-step budget with final-checkpoint evaluation, after early-stopping analysis showed that checkpoint selection could distort comparisons across gold-only and distilled settings.

Important caveat:

> The step budgets are pragmatic report-grade choices, not claims of global optimality. They prioritize comparability and reproducibility over per-run checkpoint tuning.
