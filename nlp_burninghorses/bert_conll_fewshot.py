from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List

import datasets as hf_datasets
import torch
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    set_seed,
)
from transformers.utils import logging as hf_logging

from bert_eval import (
    evaluate_model,
    make_dataloader,
    tokenize_and_align_labels_fn,
)
from bert_results import append_result_row, build_result_row

try:
    from huggingface_hub.utils import disable_progress_bars
except ImportError:
    disable_progress_bars = None


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

MODEL_NAME = "google-bert/bert-base-cased"

LEARNING_RATE = 2e-5
MAX_TRAIN_STEPS = 1000
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16

EARLY_STOPPING_EVAL_EVERY = 15
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_STEPS_BEFORE_EVAL = 150

RESULTS_FILENAME = "BERT_kshot_results.csv"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


DATA_ROOT = repo_root() / "data" / "interim" / "conll2003_kshot_bert"
MODELS_ROOT = repo_root() / "models" / "bert_conll_kshot"
RESULTS_ROOT = repo_root() / "results" / "bert_conll2003_fewshot"


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def read_metadata(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dataset_from_jsonl(path: Path) -> Dataset:
    return Dataset.from_list(read_jsonl(path))


def find_split_dirs(data_root: Path) -> List[Path]:
    return sorted(
        [p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("k")],
        key=lambda p: p.name,
    )


def csv_from_env(name: str, default: List[str]) -> List[str]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_with_early_stopping(
    model,
    train_dataloader,
    mini_eval_dataloader,
    optimizer,
    device,
    label_list,
    *,
    max_steps: int,
    eval_every: int,
    patience: int,
    min_steps_before_eval: int,
    best_model_save_dir: Path,
) -> Dict[str, Any]:
    train_iter = iter(train_dataloader)

    global_step = 0
    total_loss = 0.0

    best_span_f1 = -1.0
    best_step = 0
    patience_left = patience
    early_stopped = False
    num_evals = 0

    pbar = tqdm(total=max_steps, desc="Train", dynamic_ncols=True)
    last_loss = 0.0

    while global_step < max_steps:
        model.train()

        if global_step < min_steps_before_eval:
            steps_this_round = min(
                min_steps_before_eval - global_step,
                max_steps - global_step,
            )
        else:
            steps_this_round = min(eval_every, max_steps - global_step)

        for _ in range(steps_this_round):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)

            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            last_loss = loss.item()
            global_step += 1

            pbar.update(1)
            pbar.set_postfix_str(
                f"loss={last_loss:.3f} best={best_span_f1:.3f} pat={patience_left}"
            )

        if global_step < min_steps_before_eval:
            continue

        num_evals += 1

        _, span_metrics = evaluate_model(
            model,
            mini_eval_dataloader,
            device,
            label_list,
            show_progress=False,
        )

        span_f1_score = span_metrics["Span_Strict_F1"]

        if span_f1_score > best_span_f1:
            best_span_f1 = span_f1_score
            best_step = global_step
            patience_left = patience

            best_model_save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_model_save_dir)
        else:
            patience_left -= 1

        pbar.set_postfix_str(
            f"loss={last_loss:.3f} miniSF1={span_f1_score:.3f} "
            f"best={best_span_f1:.3f}@{best_step} pat={patience_left}"
        )

        if patience_left <= 0:
            early_stopped = True
            break

    pbar.close()

    if num_evals == 0 and global_step > 0:
        best_model_save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(best_model_save_dir)
        best_step = global_step

    avg_loss = total_loss / global_step if global_step else 0.0

    return {
        "avg_loss": avg_loss,
        "steps_trained": global_step,
        "best_span_strict_f1": best_span_f1,
        "best_step": best_step,
        "early_stopped": early_stopped,
        "num_evals": num_evals,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=".*Precision and F-score are ill-defined.*",
    )

    hf_logging.set_verbosity_error()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    if disable_progress_bars is not None:
        disable_progress_bars()

    hf_datasets.disable_progress_bars()

    results_root = Path(os.environ.get("BERT_KSHOT_RESULTS_ROOT", RESULTS_ROOT)).resolve()
    models_root = Path(os.environ.get("BERT_KSHOT_MODELS_ROOT", MODELS_ROOT)).resolve()
    results_filename = os.environ.get("BERT_KSHOT_RESULTS_FILENAME", RESULTS_FILENAME)
    requested_splits = csv_from_env("BERT_KSHOT_SPLITS", [])

    results_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    top_metadata = read_metadata(DATA_ROOT / "metadata.json")

    label_list = top_metadata["label_list"]
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for i, label in enumerate(label_list)}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    tokenize_and_align_labels = tokenize_and_align_labels_fn(
        tokenizer,
        label_to_id,
    )

    validation_ds = dataset_from_jsonl(DATA_ROOT / "validation.jsonl")
    mini_eval_ds = dataset_from_jsonl(DATA_ROOT / "mini_val.jsonl")
    test_ds = dataset_from_jsonl(DATA_ROOT / "test.jsonl")

    split_dirs = find_split_dirs(DATA_ROOT)
    if requested_splits:
        requested = set(requested_splits)
        split_dirs = [split_dir for split_dir in split_dirs if split_dir.name in requested]

    print(f"Found {len(split_dirs)} K-shot split folders")
    print(f"Mini eval size: {len(mini_eval_ds)}")
    print(f"Validation size: {len(validation_ds)}")
    print(f"Test size: {len(test_ds)}")

    shared_eval_dd = DatasetDict(
        {
            "mini_eval": mini_eval_ds,
            "validation": validation_ds,
            "test": test_ds,
        }
    )

    processed_shared_eval = shared_eval_dd.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=mini_eval_ds.column_names,
        desc="Tokenize shared eval splits",
    )

    mini_eval_dataloader = make_dataloader(
        processed_shared_eval["mini_eval"],
        tokenizer,
        EVAL_BATCH_SIZE,
    )

    validation_dataloader = make_dataloader(
        processed_shared_eval["validation"],
        tokenizer,
        EVAL_BATCH_SIZE,
    )

    test_dataloader = make_dataloader(
        processed_shared_eval["test"],
        tokenizer,
        EVAL_BATCH_SIZE,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    csv_path = results_root / results_filename

    for run_index, split_dir in enumerate(split_dirs):
        split_metadata = read_metadata(split_dir / "metadata.json")

        k_shot = int(split_metadata["k"])
        split_seed = int(split_metadata["seed"])
        split_name = split_dir.name

        model_dir = models_root / split_name
        summary_path = model_dir / "training_summary.json"
        if summary_path.exists() and os.environ.get("BERT_KSHOT_OVERWRITE", "0") != "1":
            print(f"\n=== Skipping existing {split_name}: {summary_path} ===")
            continue

        print(f"\n=== Training {split_name} ===")

        set_seed(split_seed)

        train_ds = dataset_from_jsonl(split_dir / "train.jsonl")

        processed_train = train_ds.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=train_ds.column_names,
            desc=f"Tokenize {split_name}",
        )

        train_dataloader = make_dataloader(
            processed_train,
            tokenizer,
            TRAIN_BATCH_SIZE,
            shuffle=True,
        )

        config = AutoConfig.from_pretrained(
            MODEL_NAME,
            num_labels=len(label_list),
            id2label=id_to_label,
            label2id=label_to_id,
        )

        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            config=config,
        )
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        best_ckpt_dir = model_dir / "best_span_f1"

        train_summary = train_with_early_stopping(
            model,
            train_dataloader,
            mini_eval_dataloader,
            optimizer,
            device,
            label_list,
            max_steps=MAX_TRAIN_STEPS,
            eval_every=EARLY_STOPPING_EVAL_EVERY,
            patience=EARLY_STOPPING_PATIENCE,
            min_steps_before_eval=EARLY_STOPPING_MIN_STEPS_BEFORE_EVAL,
            best_model_save_dir=best_ckpt_dir,
        )

        print(
            f"Finished {split_name}: "
            f"steps={train_summary['steps_trained']} "
            f"best mini Span Strict F1={train_summary['best_span_strict_f1']:.4f}"
        )

        model = AutoModelForTokenClassification.from_pretrained(best_ckpt_dir)
        model.to(device)

        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

        eval_loaders = {
            "mini_eval": (mini_eval_dataloader, len(mini_eval_ds)),
            "validation": (validation_dataloader, len(validation_ds)),
            "test": (test_dataloader, len(test_ds)),
        }
        eval_results: Dict[str, Dict[str, float]] = {}

        for eval_split, (loader, n_eval) in eval_loaders.items():
            token_metrics, span_metrics = evaluate_model(
                model,
                loader,
                device,
                label_list,
                show_progress=False,
            )

            row = build_result_row(
                split_name=split_name,
                k_shot=k_shot,
                split_seed=split_seed,
                run_index=run_index,
                eval_split=eval_split,
                n_train=len(train_ds),
                n_eval=n_eval,
                max_train_steps=MAX_TRAIN_STEPS,
                train_batch_size=TRAIN_BATCH_SIZE,
                eval_batch_size=EVAL_BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                train_summary=train_summary,
                token_metrics=token_metrics,
                span_metrics=span_metrics,
            )

            append_result_row(csv_path, row)
            eval_results[eval_split] = {**token_metrics, **span_metrics}

            print(
                f"{eval_split}: "
                f"HF F1={token_metrics['HF_Token_F1']:.4f} | "
                f"Span strict F1={span_metrics['Span_Strict_F1']:.4f}"
            )

        summary = {
            "experiment": {
                "dataset": "conll2003",
                "model_name": MODEL_NAME,
                "split": split_name,
                "k_shot": k_shot,
                "split_seed": split_seed,
                "seed": split_seed,
                "note": "Standard BERT k-shot baseline with shared mini-val selection and validation/test scoring.",
            },
            "paths": {
                "train_file": str(split_dir / "train.jsonl"),
                "mini_eval_file": str(DATA_ROOT / "mini_val.jsonl"),
                "validation_file": str(DATA_ROOT / "validation.jsonl"),
                "test_file": str(DATA_ROOT / "test.jsonl"),
                "output_dir": str(model_dir),
                "results_csv": str(csv_path),
            },
            "runtime_config": {
                "max_steps": MAX_TRAIN_STEPS,
                "eval_every": EARLY_STOPPING_EVAL_EVERY,
                "patience": EARLY_STOPPING_PATIENCE,
                "min_steps_before_eval": EARLY_STOPPING_MIN_STEPS_BEFORE_EVAL,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
            },
            "data_stats": {
                "train_rows": len(train_ds),
                "mini_eval_rows": len(mini_eval_ds),
                "validation_rows": len(validation_ds),
                "test_rows": len(test_ds),
            },
            "train_summary": train_summary,
            "eval_results": eval_results,
        }
        write_json(summary_path, summary)

    print(f"\nDone. Results written to: {csv_path}")


if __name__ == "__main__":
    main()