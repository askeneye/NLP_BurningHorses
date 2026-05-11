from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List


CSV_FIELDNAMES = [
    "Split_Name",
    "K_shot",
    "Split_Seed",
    "Run_Index",
    "Eval_Split",
    "N_Train_Sentences",
    "N_Eval_Sentences",
    "Max_Train_Steps",
    "Train_Steps",
    "Best_Span_F1_Step",
    "Early_Stopped",
    "Train_Batch_Size",
    "Eval_Batch_Size",
    "Learning_Rate",
    "HF_Token_Precision",
    "HF_Token_Recall",
    "HF_Token_F1",
    "HF_Token_Accuracy",
    "Span_Strict_Precision",
    "Span_Strict_Recall",
    "Span_Strict_F1",
    "Span_Unlabeled_Precision",
    "Span_Unlabeled_Recall",
    "Span_Unlabeled_F1",
    "Span_Loose_Precision",
    "Span_Loose_Recall",
    "Span_Loose_F1",
]


def build_result_row(
    *,
    split_name: str,
    k_shot: int,
    split_seed: int,
    run_index: int,
    eval_split: str,
    n_train: int,
    n_eval: int,
    max_train_steps: int,
    train_batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    train_summary: Dict[str, Any],
    token_metrics: Dict[str, float],
    span_metrics: Dict[str, float],
) -> Dict[str, Any]:
    row = {
        "Split_Name": split_name,
        "K_shot": k_shot,
        "Split_Seed": split_seed,
        "Run_Index": run_index,
        "Eval_Split": eval_split,
        "N_Train_Sentences": n_train,
        "N_Eval_Sentences": n_eval,
        "Max_Train_Steps": max_train_steps,
        "Train_Steps": train_summary["steps_trained"],
        "Best_Span_F1_Step": train_summary["best_step"],
        "Early_Stopped": int(train_summary["early_stopped"]),
        "Train_Batch_Size": train_batch_size,
        "Eval_Batch_Size": eval_batch_size,
        "Learning_Rate": learning_rate,
    }

    row.update(token_metrics)
    row.update(span_metrics)

    return row


def ensure_csv_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and csv_path.stat().st_size > 0:
        return

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()


def append_result_row(csv_path: Path, row: Dict[str, Any]) -> None:
    ensure_csv_header(csv_path)

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDNAMES})


def append_result_rows(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_csv_header(csv_path)

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)

        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDNAMES})