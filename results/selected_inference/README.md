# Selected inference package

Hey Andrea.

See if you can spot somthing interesting in these data.
It would be nice to know how the BART single teacher and the BERT student predeicts different entity types.
Maybe do some plot and do a few visual inspections, if there is a pattern worth reflecting on.

Also i would be nice to just do a sanity check on the k=5 splits, and see if there are any weird input that would affect the fewshot experiments.

This folder packages simple label-level artifacts for a report side analysis.

Selection: the reported `pat_perm_bart_ensemble_to_bert` model at `k=5` uses the median test split `k5_seed242` by strict span F1 across seeds 142, 242, and 42.

Useful files:

- `bert_student_ensemble_hard_argmax_k5_seed242_test_predictions_eval128.jsonl`: exported predictions for the reported BERT student using the same `max_length=128` setting as the report metrics.
- `bert_student_ensemble_hard_argmax_k5_seed242_test_predictions_max512.jsonl`: full-length BERT student predictions for qualitative inspection.
- `bart_single_pattern_01_k5_seed242_test_predictions.jsonl`: comparable single BART learner predictions.
- `bart_ensemble_hard_argmax_k5_seed242_test_predictions.jsonl`: comparable BART ensemble hard-vote predictions.
- `span_strict_stats_by_entity_type.csv`: exact-span precision, recall, and F1 by entity type.
- `token_label_precision_recall.csv`: BIO-label token precision, recall, and F1.
- `selection_summary.json`: machine-readable description of the selection and file contents.

Each prediction JSONL row contains `tokens`, `gold_tags`, and `predicted_tags`.
For the BERT `eval128` file, long sentences are truncated to the words actually evaluated by the report configuration.
