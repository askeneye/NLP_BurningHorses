from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start_path: Path) -> Path:
    for path in [start_path, *start_path.parents]:
        if (path / "pyproject.toml").exists():
            return path
    raise FileNotFoundError("Could not locate project root containing pyproject.toml.")


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
DEFAULT_INPUT_FILE = (
    "data/interim/conll2003_ensemble_teachers/bart_base/pet_oada_verbalized_final2000/"
    "k5_seed42/validation/ensemble_hard_argmax.jsonl"
)
DEFAULT_OUTPUT_FILE = "reports/ensemble_tagging/k5_seed42_validation_metrics.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)


def to_spans(tags: list[str]) -> set[tuple[int, int, str]]:
    spans: set[tuple[int, int, str]] = set()
    start: int | None = None
    entity_type: str | None = None
    for index, tag in enumerate([*tags, "O"]):
        if tag == "O" or "-" not in tag:
            if start is not None and entity_type is not None:
                spans.add((start, index, entity_type))
            start = None
            entity_type = None
            continue
        prefix, current_type = tag.split("-", 1)
        if prefix == "B" or entity_type != current_type:
            if start is not None and entity_type is not None:
                spans.add((start, index, entity_type))
            start = index
            entity_type = current_type
    return spans


def strict_span_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    tp = fp = fn = 0
    token_correct = 0
    token_total = 0
    for row in rows:
        gold_tags = row.get("gold_tags")
        predicted_tags = row.get("predicted_tags")
        if gold_tags is None or predicted_tags is None:
            raise ValueError("Rows must include gold_tags and predicted_tags.")
        if len(gold_tags) != len(predicted_tags):
            raise ValueError(f"Tag length mismatch for row {row.get('id')}.")

        token_total += len(gold_tags)
        token_correct += sum(1 for gold, pred in zip(gold_tags, predicted_tags) if gold == pred)
        gold_spans = to_spans(gold_tags)
        pred_spans = to_spans(predicted_tags)
        overlap = len(gold_spans.intersection(pred_spans))
        tp += overlap
        fp += len(pred_spans) - overlap
        fn += len(gold_spans) - overlap

    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "Token_Accuracy": token_correct / token_total if token_total else 0.0,
        "Span_Strict_Precision": precision,
        "Span_Strict_Recall": recall,
        "Span_Strict_F1": f1,
        "rows": len(rows),
        "tokens": token_total,
    }


def main() -> None:
    input_file = Path(os.environ.get("ENSEMBLE_EVAL_INPUT_FILE", PROJECT_ROOT / DEFAULT_INPUT_FILE)).resolve()
    output_file = Path(os.environ.get("ENSEMBLE_EVAL_OUTPUT_FILE", PROJECT_ROOT / DEFAULT_OUTPUT_FILE)).resolve()
    rows = read_jsonl(input_file)
    metrics = strict_span_metrics(rows)
    summary = {
        "input_file": str(input_file),
        "metrics": metrics,
    }
    write_json(output_file, summary)
    print(
        "Ensemble tagging eval: "
        f"rows={metrics['rows']} strict_f1={metrics['Span_Strict_F1']:.4f} output={output_file}"
    )


if __name__ == "__main__":
    main()
