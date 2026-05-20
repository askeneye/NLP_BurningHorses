from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any


SELECTED_SPLIT = "k5_seed242"
ENTITY_TYPES = ["PER", "ORG", "LOC", "MISC"]
LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
OUT_DIR = Path(__file__).resolve().parent

BERT_MODEL_DIR = (
    ROOT
    / "models"
    / "bert_conll_distilled_fixed_final600_report"
    / SELECTED_SPLIT
    / "pat_perm_bart_ensemble_to_bert"
    / "hard_argmax"
)
TEST_FILE = ROOT / "data" / "interim" / "conll2003_kshot_bert" / "test.jsonl"
BART_SINGLE_PREDICTIONS = (
    ROOT
    / "data"
    / "interim"
    / "conll2003_soft_labels"
    / "bart_base"
    / "pet_oada_verbalized"
    / SELECTED_SPLIT
    / "pattern_01_xe_verbalized_final2000"
    / "test"
    / "hard_predictions.jsonl"
)
BART_SINGLE_METRICS = (
    ROOT / "reports" / "single_bart_tagging" / f"{SELECTED_SPLIT}_pat_perm_bart_pattern_01_test_metrics.json"
)
BART_ENSEMBLE_PREDICTIONS = (
    ROOT
    / "data"
    / "interim"
    / "conll2003_ensemble_teachers"
    / "bart_base"
    / "pet_oada_verbalized_final2000"
    / SELECTED_SPLIT
    / "test"
    / "ensemble_hard_argmax.jsonl"
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_spans(tags: list[str]) -> set[tuple[int, int, str]]:
    # Match nlp_burninghorses/utils/span_f1.py, which defines a span from each
    # B-* tag through the following contiguous I-* run.
    spans: set[tuple[int, int, str]] = set()
    for start, tag in enumerate(tags):
        if not tag.startswith("B-"):
            continue
        end = start
        for end in range(start + 1, len(tags)):
            if not tags[end].startswith("I-"):
                break
        spans.add((start, end, tag[2:]))
    return spans


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def span_stats(model_name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gold_by_type: dict[str, set[tuple[int, int, str, int]]] = {entity: set() for entity in ENTITY_TYPES}
    pred_by_type: dict[str, set[tuple[int, int, str, int]]] = {entity: set() for entity in ENTITY_TYPES}

    for row_index, row in enumerate(rows):
        for start, end, entity_type in to_spans(row["gold_tags"]):
            gold_by_type[entity_type].add((start, end, entity_type, row_index))
        for start, end, entity_type in to_spans(row["predicted_tags"]):
            pred_by_type[entity_type].add((start, end, entity_type, row_index))

    stats: list[dict[str, Any]] = []
    total_gold = total_pred = total_tp = 0
    for entity_type in ENTITY_TYPES:
        gold_spans = gold_by_type[entity_type]
        pred_spans = pred_by_type[entity_type]
        true_positives = len(gold_spans & pred_spans)
        precision = safe_divide(true_positives, len(pred_spans))
        recall = safe_divide(true_positives, len(gold_spans))
        stats.append(
            {
                "model": model_name,
                "entity_type": entity_type,
                "gold_spans": len(gold_spans),
                "predicted_spans": len(pred_spans),
                "true_positive_spans": true_positives,
                "precision": precision,
                "recall": recall,
                "f1": f1(precision, recall),
            }
        )
        total_gold += len(gold_spans)
        total_pred += len(pred_spans)
        total_tp += true_positives

    precision = safe_divide(total_tp, total_pred)
    recall = safe_divide(total_tp, total_gold)
    stats.append(
        {
            "model": model_name,
            "entity_type": "ALL",
            "gold_spans": total_gold,
            "predicted_spans": total_pred,
            "true_positive_spans": total_tp,
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
        }
    )
    return stats


def token_label_recall(model_name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for label in LABELS:
        gold_count = 0
        correct_count = 0
        predicted_count = 0
        for row in rows:
            for gold, predicted in zip(row["gold_tags"], row["predicted_tags"]):
                if gold == label:
                    gold_count += 1
                    if predicted == label:
                        correct_count += 1
                if predicted == label:
                    predicted_count += 1
        precision = safe_divide(correct_count, predicted_count)
        recall = safe_divide(correct_count, gold_count)
        stats.append(
            {
                "model": model_name,
                "label": label,
                "gold_tokens": gold_count,
                "predicted_tokens": predicted_count,
                "correct_tokens": correct_count,
                "precision": precision,
                "recall": recall,
                "f1": f1(precision, recall),
            }
        )
    return stats


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_bert_predictions(output_file: Path, *, max_length: int) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    test_rows = read_jsonl(TEST_FILE)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(BERT_MODEL_DIR)
    model.eval()

    id_to_label = {int(key): value for key, value in model.config.id2label.items()}
    batch_size = 32
    prediction_rows: list[dict[str, Any]] = []

    for start in range(0, len(test_rows), batch_size):
        batch_rows = test_rows[start : start + batch_size]
        encoded = tokenizer(
            [row["tokens"] for row in batch_rows],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**encoded).logits
        predicted_ids = logits.argmax(dim=-1).tolist()

        for batch_index, row in enumerate(batch_rows):
            predicted_tags: list[str | None] = [None] * len(row["tokens"])
            for token_index, word_index in enumerate(encoded.word_ids(batch_index=batch_index)):
                if word_index is not None and predicted_tags[word_index] is None:
                    predicted_tags[word_index] = id_to_label[predicted_ids[batch_index][token_index]]

            labelled_word_count = sum(tag is not None for tag in predicted_tags)
            if labelled_word_count == 0:
                raise ValueError(f"Sentence {start + batch_index} produced no labelled words.")
            predicted_tags = predicted_tags[:labelled_word_count]

            prediction_rows.append(
                {
                    "id": f"row-{start + batch_index:06d}",
                    "tokens": row["tokens"][:labelled_word_count],
                    "ner_tags": row["ner_tags"][:labelled_word_count],
                    "gold_tags": [id_to_label[label_id] for label_id in row["ner_tags"][:labelled_word_count]],
                    "predicted_tags": predicted_tags,
                    "max_length": max_length,
                    "original_token_count": len(row["tokens"]),
                    "truncated": labelled_word_count < len(row["tokens"]),
                }
            )

    write_jsonl(output_file, prediction_rows)
    return prediction_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    copied_files = {
        "bart_single_pattern_01_k5_seed242_test_predictions.jsonl": BART_SINGLE_PREDICTIONS,
        "bart_single_pattern_01_k5_seed242_test_metrics.json": BART_SINGLE_METRICS,
        "bart_ensemble_hard_argmax_k5_seed242_test_predictions.jsonl": BART_ENSEMBLE_PREDICTIONS,
        "bert_student_ensemble_hard_argmax_k5_seed242_training_summary.json": BERT_MODEL_DIR
        / "training_summary.json",
    }
    for output_name, source_path in copied_files.items():
        shutil.copy2(source_path, OUT_DIR / output_name)

    bert_predictions_eval128 = export_bert_predictions(
        OUT_DIR / "bert_student_ensemble_hard_argmax_k5_seed242_test_predictions_eval128.jsonl",
        max_length=128,
    )
    export_bert_predictions(
        OUT_DIR / "bert_student_ensemble_hard_argmax_k5_seed242_test_predictions_max512.jsonl",
        max_length=512,
    )
    models = {
        "bert_student_ensemble_hard_argmax_eval128": bert_predictions_eval128,
        "bart_single_pattern_01": read_jsonl(BART_SINGLE_PREDICTIONS),
        "bart_ensemble_hard_argmax": read_jsonl(BART_ENSEMBLE_PREDICTIONS),
    }

    span_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    for model_name, rows in models.items():
        span_rows.extend(span_stats(model_name, rows))
        token_rows.extend(token_label_recall(model_name, rows))

    write_csv(OUT_DIR / "span_strict_stats_by_entity_type.csv", span_rows)
    write_csv(OUT_DIR / "token_label_precision_recall.csv", token_rows)

    summary = {
        "selection": {
            "reported_model": "pat_perm_bart_ensemble_to_bert",
            "k_shot": 5,
            "median_split": SELECTED_SPLIT,
            "median_split_basis": {
                "k5_seed142_test_span_strict_f1": 0.4570973521799721,
                "k5_seed242_test_span_strict_f1": 0.5717602363866611,
                "k5_seed42_test_span_strict_f1": 0.6639934384398067,
            },
        },
        "files": {
            "bert_student_predictions_eval128": "bert_student_ensemble_hard_argmax_k5_seed242_test_predictions_eval128.jsonl",
            "bert_student_predictions_max512": "bert_student_ensemble_hard_argmax_k5_seed242_test_predictions_max512.jsonl",
            "bart_single_predictions": "bart_single_pattern_01_k5_seed242_test_predictions.jsonl",
            "bart_ensemble_predictions": "bart_ensemble_hard_argmax_k5_seed242_test_predictions.jsonl",
            "span_stats": "span_strict_stats_by_entity_type.csv",
            "token_stats": "token_label_precision_recall.csv",
        },
        "row_format": {
            "tokens": "Original CoNLL tokens.",
            "gold_tags": "Gold BIO labels.",
            "predicted_tags": "Model output BIO labels.",
        },
    }
    write_json(OUT_DIR / "selection_summary.json", summary)

    readme = """# Selected inference package

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
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
