# Curated Data Dictionary

Curated CSVs are the handover contract between experiment-running agents and report/plotting agents.

Each row should describe one measured value or one plotted aggregate. Keep raw logs, JSONL files, and training artifacts in their original locations; record their paths in `source_path`.

## Glossary

- `Orders`: Ordering instructions in the input data. Use `Orders` when discussing the prompts or training examples that tell the model which entity order to follow.
- `Patterns`: Prompt templates used to verbalize the NER task for BART.
- `Verbalized`: Data or predictions represented as generated text rather than token-level BIO labels.
- `Verbalizer`: The mapping/convention that turns NER structures into generated text and back again.
- `soft_labels`: Ensemble vote distributions intended for distillation. These were explored but shelved for the current report direction unless explicitly revived.

## Required Fields

Use these fields for curated experiment-result CSVs:

- `dataset`: Dataset id, for example `conll2003` or `fewnerd`.
- `task`: Task family, for example `ner`.
- `label_level`: Label granularity, for example `coarse`, `fine`, or blank if not applicable.
- `tagging_scheme`: Tagging format, for example `BIO`, or blank if not applicable.
- `method_id`: Stable machine-readable method id, for example `bert_baseline`, `orders_pattern_ensemble_majority`, or `orders_pattern_ensemble_distilled_bert`.
- `method_label`: Display label from `styleguide.md`.
- `model`: Base model or architecture, for example `bert-base-cased` or `facebook/bart-base`.
- `k_shot`: k-shot setting.
- `split_seed`: Dataset split seed.
- `run_seed`: Training/model seed if known.
- `run_id`: Optional run identifier or experiment folder name.
- `pattern_id`: Prompt pattern id if applicable, otherwise blank.
- `order_family`: Ordering-instruction family if applicable, for example `orders` or blank.
- `ensemble_size`: Number of ensemble members if applicable, otherwise blank.
- `teacher_source`: Teacher data source for distillation, for example `orders_pattern_ensemble_majority`.
- `eval_split`: Evaluation split, for example `mini_val`, `validation`, or `test`.
- `metric_name`: Metric id, for example `span_strict_f1`.
- `metric_label`: Human-readable metric name, for example `Strict span F1`.
- `metric_value`: Metric value, usually on a `0.0` to `1.0` scale.
- `metric_unit`: Usually `fraction`; use `count` for counts.
- `status`: One of the status values below.
- `score_source`: One of the score-source values below.
- `is_placeholder`: `true` if the row is temporary design data.
- `needs_more_runs`: `true` if the row is incomplete across seeds, k values, or patterns.
- `expected_runs`: Expected number of runs for this aggregate.
- `completed_runs`: Completed number of runs represented by this row.
- `source_path`: Repo-relative path to the source artifact.
- `notes`: Short explanation of limitations or interpretation.
- `last_updated`: ISO date, for example `2026-05-17`.

## Status Values

- `final`: Complete and intended for final report claims. Safe for final figures.
- `in_progress`: Valid result but more runs are expected. Use only with a clear caveat.
- `placeholder`: Temporary design value, often validation or manual mock data. Use only for layout/design drafts.
- `needs_rerun`: Known issue or configuration mismatch means the score should be replaced. Do not use for claims.
- `superseded`: Kept for traceability but replaced by newer data. Do not use in current figures.

## Score Source Values

- `test`: Final held-out test score.
- `validation`: Validation score from the main validation split.
- `mini_validation`: Small validation or mini-eval score used during training.
- `pilot_validation`: Validation score from pilot experiments not intended as final evidence.
- `manual_placeholder`: Manually entered value for plot design only.
- `aggregate`: Derived aggregate such as mean, standard deviation, or confidence interval.

## Metric Names

Prefer these metric ids:

- `span_strict_precision`
- `span_strict_recall`
- `span_strict_f1`
- `span_unlabeled_f1`
- `span_loose_f1`
- `token_precision`
- `token_recall`
- `token_f1`
- `token_accuracy`
- `entity_mentions`
- `row_jaccard`
- `token_agreement`

## Handover Checklist

Before handing over a curated CSV:

1. Every row has `status`, `score_source`, `is_placeholder`, and `needs_more_runs`.
2. Validation and pilot scores are not labeled as final test scores.
3. `method_label` follows `styleguide.md`.
4. `source_path` points to the raw artifact or original result table.
5. If runs are incomplete, `expected_runs` and `completed_runs` make that visible.
