from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

from scripts._lib.repro_io import read_json_lines, write_json


def resolve_step07_paths(root: Path, config: dict[str, Any]) -> dict[str, Path | str]:
    split_name = str(config.get("default_profile", {}).get("split_name", "k5_seed242"))
    paths = config.get("paths", {})
    interim_root = root / str(paths.get("interim_root", "reproduction/data/interim"))
    models_root = root / str(paths.get("models_root", "reproduction/models"))
    reports_root = root / str(paths.get("reports_root", "reproduction/reports"))
    logs_root = root / str(paths.get("logs_root", "reproduction/logs"))
    return {
        "split_name": split_name,
        "base_root": interim_root / split_name / "base",
        "student_root": models_root / "student" / split_name,
        "reports_root": reports_root / "student" / split_name,
        "logs_root": logs_root,
        "step06_manifest": logs_root / "step06_train_bert_manifest.json",
        "step07_manifest": logs_root / "step07_score_bert_manifest.json",
    }


def _bio_spans(tags: list[str]) -> set[tuple[int, int, str]]:
    spans: set[tuple[int, int, str]] = set()
    start = -1
    ent_type = ""
    for index, tag in enumerate(tags):
        if tag == "O":
            if start >= 0:
                spans.add((start, index, ent_type))
            start = -1
            ent_type = ""
            continue
        prefix, current_type = tag.split("-", 1) if "-" in tag else ("O", "")
        if prefix == "B" or (start >= 0 and current_type != ent_type):
            if start >= 0:
                spans.add((start, index, ent_type))
            start = index
            ent_type = current_type
        elif start < 0:
            start = index
            ent_type = current_type
    if start >= 0:
        spans.add((start, len(tags), ent_type))
    return spans


def _span_strict_f1(gold_sequences: list[list[str]], pred_sequences: list[list[str]]) -> dict[str, float]:
    tp = fp = fn = 0
    for gold_tags, pred_tags in zip(gold_sequences, pred_sequences):
        gold_spans = _bio_spans(gold_tags)
        pred_spans = _bio_spans(pred_tags)
        overlap = len(gold_spans.intersection(pred_spans))
        tp += overlap
        fp += len(pred_spans) - overlap
        fn += len(gold_spans) - overlap
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"Span_Strict_Precision": precision, "Span_Strict_Recall": recall, "Span_Strict_F1": f1}


def _build_eval_dataset(rows: list[dict[str, Any]], tokenizer, label_to_id: dict[str, int], max_length: int) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for row in rows:
        tokenized = tokenizer(
            row["tokens"],
            max_length=max_length,
            truncation=True,
            is_split_into_words=True,
        )
        word_ids = tokenized.word_ids()
        labels: list[int] = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None or word_id == previous_word_id:
                labels.append(-100)
            else:
                labels.append(label_to_id[row["gold_tags"][word_id]])
            previous_word_id = word_id
        features.append(
            {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": labels,
                "tokens": row["tokens"],
                "gold_tags": row["gold_tags"],
            }
        )
    return features


def score_student_model(
    *,
    base_root: Path,
    model_dir: Path,
    max_length: int = 128,
    batch_size: int = 64,
) -> dict[str, Any]:
    metadata = json.loads((base_root / "metadata.json").read_text(encoding="utf-8"))
    label_list = [str(label) for label in metadata["label_list"]]
    label_to_id = {label: index for index, label in enumerate(label_list)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    split_metrics: dict[str, Any] = {}
    for split_name in ["test"]:
        rows = read_json_lines(base_root / f"{split_name}.jsonl")
        rows_with_gold = [
            {"tokens": row["tokens"], "gold_tags": [label_list[int(tag)] for tag in row["ner_tags"]]}
            for row in rows
        ]
        features = _build_eval_dataset(rows_with_gold, tokenizer, label_to_id, max_length)
        token_features = [{"input_ids": feat["input_ids"], "attention_mask": feat["attention_mask"], "labels": feat["labels"]} for feat in features]
        dataloader = DataLoader(
            token_features,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DataCollatorForTokenClassification(tokenizer),
        )

        pred_sequences: list[list[str]] = []
        gold_sequences: list[list[str]] = []
        token_index = 0
        with torch.no_grad():
            for batch in dataloader:
                labels = batch["labels"]
                inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
                logits = model(**inputs).logits.detach().cpu()
                predictions = logits.argmax(dim=-1)
                for row_idx in range(predictions.shape[0]):
                    feat = features[token_index]
                    token_index += 1
                    word_tags: list[str] = []
                    word_golds: list[str] = []
                    for pos in range(predictions.shape[1]):
                        if labels[row_idx, pos].item() == -100:
                            continue
                        word_tags.append(id_to_label[int(predictions[row_idx, pos].item())])
                        word_golds.append(id_to_label[int(labels[row_idx, pos].item())])
                    pred_sequences.append(word_tags)
                    gold_sequences.append(word_golds)
        split_metrics[split_name] = _span_strict_f1(gold_sequences, pred_sequences)
    return split_metrics


def write_step07_outputs(
    *,
    reports_root: Path,
    manifest_path: Path,
    split_name: str,
    teacher_variant: str,
    run_tag: str,
    model_dir: Path,
    metrics: dict[str, Any],
) -> tuple[Path, Path]:
    report_path = reports_root / f"{teacher_variant}__{run_tag}_metrics.json"
    payload = {
        "split_name": split_name,
        "teacher_variant": teacher_variant,
        "run_tag": run_tag,
        "model_dir": str(model_dir),
        "metrics": metrics,
    }
    write_json(report_path, payload)
    write_json(manifest_path, {"report_file": str(report_path), **payload})
    return report_path, manifest_path


def print_step07_summary(report_path: Path, manifest_path: Path, metrics: dict[str, Any]) -> None:
    print("Scored distilled BERT student")
    print(f"test_strict_f1={metrics['test']['Span_Strict_F1']:.4f}")
    print(f"report_file={report_path}")
    print(f"run_manifest={manifest_path}")
