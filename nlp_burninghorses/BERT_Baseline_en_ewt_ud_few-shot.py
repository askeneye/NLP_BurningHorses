# English EWT (Universal NER): greedy K-shot train split (Yang & Katiyar 2020, Alg. 1),
# full dev for evaluation. Early stopping on dev Span_Strict_F1 every N steps after a warmup (not token accuracy).
# Multiple K values, 3 runs per K (different split seeds), mean row per K.
# From repo root: ./venv/bin/python nlp_burninghorses/BERT_Baseline_en_ewt_ud_few-shot.py

from __future__ import annotations

import csv
import importlib.util
import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import datasets as hf_datasets
import evaluate
import span_f1
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    set_seed,
)
from transformers.utils import logging as hf_logging

try:
    from huggingface_hub.utils import disable_progress_bars
except ImportError:
    disable_progress_bars = None  # type: ignore[misc, assignment]

# ----------------------------------------------------------------------------
# Paths & K-split loader
# ----------------------------------------------------------------------------


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_k_split_module() -> Any:
    path = Path(__file__).resolve().parent / "K-split.py"
    spec = importlib.util.spec_from_file_location("k_split_impl", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load K-split from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_en_ewt_ud_iob2(path: str):
    """Read Universal NER IOB2: tab lines index, token, NER, …; # comments and blank lines separate sentences."""
    sentences_tokens, sentences_tags, label_set = [], [], set()
    current_tokens, current_tags = [], []

    def flush():
        nonlocal current_tokens, current_tags
        if current_tokens:
            sentences_tokens.append(current_tokens)
            sentences_tags.append(current_tags)
            current_tokens, current_tags = [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                flush()
                continue
            if not line.strip():
                flush()
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            current_tokens.append(parts[1])
            current_tags.append(parts[2])
            label_set.add(parts[2])
        flush()
    return sentences_tokens, sentences_tags, label_set


def load_en_ewt_ud_iob2_datasets(root: str) -> DatasetDict:
    train_path = os.path.join(root, "data", "raw", "en_ewt-ud-train.iob2")
    dev_path = os.path.join(root, "data", "raw", "en_ewt-ud-dev.iob2")
    tr_tok, tr_tag, s_train = parse_en_ewt_ud_iob2(train_path)
    dv_tok, dv_tag, s_dev = parse_en_ewt_ud_iob2(dev_path)
    label_names = sorted(s_train | s_dev)
    if "O" in label_names:
        label_names.remove("O")
        label_list = ["O"] + label_names
    else:
        label_list = label_names
    label_to_id = {l: i for i, l in enumerate(label_list)}

    def encode_tags(tag_seqs):
        return [[label_to_id[t] for t in sent] for sent in tag_seqs]

    features = Features(
        {
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=label_list)),
        }
    )
    train_ds = Dataset.from_dict(
        {"tokens": tr_tok, "ner_tags": encode_tags(tr_tag)},
        features=features,
    )
    val_ds = Dataset.from_dict(
        {"tokens": dv_tok, "ner_tags": encode_tags(dv_tag)},
        features=features,
    )
    return DatasetDict(train=train_ds, validation=val_ds)


# ----------------------------------------------------------------------------
# Hyperparameters (few-shot)
# ----------------------------------------------------------------------------

learning_rate = 2e-5
model_name = "google-bert/bert-base-cased"
max_train_steps = 1000
train_batch_size = 16
eval_batch_size = 16

# Early stopping: optimize span-level strict F1 (not token accuracy).
early_stopping_eval_every = 15
early_stopping_patience = 5  # stop after this many evals without a new best Span_Strict_F1
early_stopping_min_steps_before_eval = 150  # no dev eval / patience until this many train steps

K_VALUES = [5,10, 20, 50]
NUM_RUNS_PER_K = 3
# Split seeds for greedy K-shot support sampling (one per run; same triple for every K).
SPLIT_SEEDS = [42, 142, 242]

results_filename = "BERT_en_ewt_ud_fewshot_results.csv"

text_column_name = "tokens"
label_column_name = "ner_tags"


def span_metrics_from_aligned_word_tags(
    gold_tags_per_sent: List[List[str]],
    pred_tags_per_sent: List[List[str]],
) -> Dict[str, float]:
    """
    Same span aggregation as span_scores_from_files, but from aligned word-level BIO tag lists.
    """
    tp = fp = fn = 0
    recall_loose_tp = recall_loose_fn = 0
    precision_loose_tp = precision_loose_fp = 0
    tp_ul = fp_ul = fn_ul = 0

    for gold_ner, pred_ner in zip(gold_tags_per_sent, pred_tags_per_sent):
        gold_spans = span_f1.toSpans(gold_ner)
        pred_spans = span_f1.toSpans(pred_ner)
        overlap = len(gold_spans.intersection(pred_spans))
        tp += overlap
        fp += len(pred_spans) - overlap
        fn += len(gold_spans) - overlap

        overlap_ul = span_f1.getUnlabeled(gold_spans, pred_spans)
        tp_ul += overlap_ul
        fp_ul += len(pred_spans) - overlap_ul
        fn_ul += len(gold_spans) - overlap_ul

        recall_loose_tp += span_f1.getLooseOverlap(gold_spans, pred_spans)
        recall_loose_fn += len(gold_spans) - span_f1.getLooseOverlap(gold_spans, pred_spans)

        precision_loose_tp += span_f1.getLooseOverlap(pred_spans, gold_spans)
        precision_loose_fp += len(pred_spans) - span_f1.getLooseOverlap(pred_spans, gold_spans)

    prec = 0.0 if tp + fp == 0 else tp / (tp + fp)
    rec = 0.0 if tp + fn == 0 else tp / (tp + fn)
    strict_f1 = 0.0 if prec + rec == 0.0 else 2 * (prec * rec) / (prec + rec)

    prec_ul = 0.0 if tp_ul + fp_ul == 0 else tp_ul / (tp_ul + fp_ul)
    rec_ul = 0.0 if tp_ul + fn_ul == 0 else tp_ul / (tp_ul + fn_ul)
    ul_f1 = 0.0 if prec_ul + rec_ul == 0.0 else 2 * (prec_ul * rec_ul) / (prec_ul + rec_ul)

    prec_l = (
        0.0
        if precision_loose_tp + precision_loose_fp == 0
        else precision_loose_tp / (precision_loose_tp + precision_loose_fp)
    )
    rec_l = (
        0.0
        if recall_loose_tp + recall_loose_fn == 0
        else recall_loose_tp / (recall_loose_tp + recall_loose_fn)
    )
    loose_f1 = 0.0 if prec_l + rec_l == 0.0 else 2 * (prec_l * rec_l) / (prec_l + rec_l)

    return {
        "Span_Strict_Precision": prec,
        "Span_Strict_Recall": rec,
        "Span_Strict_F1": strict_f1,
        "Span_Unlabeled_Precision": prec_ul,
        "Span_Unlabeled_Recall": rec_ul,
        "Span_Unlabeled_F1": ul_f1,
        "Span_Loose_Precision": prec_l,
        "Span_Loose_Recall": rec_l,
        "Span_Loose_F1": loose_f1,
    }


def train_with_early_stopping(
    model: AutoModelForTokenClassification,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    label_list: List[str],
    metric,
    *,
    max_steps: int,
    eval_every: int,
    patience: int,
    min_steps_before_eval: int,
    best_model_save_dir: str,
) -> Dict[str, Any]:
    """
    Train up to ``max_steps``. Until ``min_steps_before_eval`` train steps, no dev eval.
    After that, every ``eval_every`` steps run dev eval on span-level strict F1;
    new best -> save weights and reset patience; else decrement patience.
    Stop when patience hits 0 or ``max_steps`` is reached.
    """
    train_iter = iter(train_dataloader)
    global_step = 0
    total_loss = 0.0
    best_span_f1 = -1.0
    best_step = 0
    patience_left = patience
    early_stopped = False
    num_evals = 0

    pbar = tqdm(
        total=max_steps,
        desc="Train",
        dynamic_ncols=True,
        mininterval=0.5,
    )
    last_loss = 0.0

    def _best_postfix() -> str:
        if best_step <= 0:
            return "best=—"
        return f"best={best_span_f1:.3f}@{best_step}"

    while global_step < max_steps:
        model.train()
        if global_step < min_steps_before_eval:
            steps_this_round = min(
                min_steps_before_eval - global_step,
                max_steps - global_step,
            )
        else:
            steps_this_round = min(eval_every, max_steps - global_step)

        if steps_this_round <= 0:
            break

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
            if global_step < min_steps_before_eval:
                pbar.set_postfix_str(
                    f"loss={last_loss:.3f} warmup {global_step}/{min_steps_before_eval}",
                    refresh=False,
                )
            else:
                pbar.set_postfix_str(
                    f"loss={last_loss:.3f} {_best_postfix()} pat={patience_left}",
                    refresh=False,
                )

        if global_step < min_steps_before_eval:
            pbar.set_postfix_str(
                f"loss={last_loss:.3f} warmup {global_step}/{min_steps_before_eval}",
                refresh=True,
            )
            continue

        num_evals += 1
        _, pred_tags, true_tags = evaluate_model(
            model,
            eval_dataloader,
            device,
            label_list,
            metric,
            show_progress=False,
        )
        span_m = span_metrics_from_aligned_word_tags(true_tags, pred_tags)
        span_f1_score = span_m["Span_Strict_F1"]

        if span_f1_score > best_span_f1:
            best_span_f1 = span_f1_score
            best_step = global_step
            patience_left = patience
            os.makedirs(best_model_save_dir, exist_ok=True)
            model.save_pretrained(best_model_save_dir)
        else:
            patience_left -= 1

        pbar.set_postfix_str(
            f"loss={last_loss:.3f} devSF1={span_f1_score:.3f} {_best_postfix()} pat={patience_left}",
            refresh=True,
        )

        if patience_left <= 0:
            early_stopped = True
            break

    pbar.close()
    if num_evals == 0 and global_step > 0:
        os.makedirs(best_model_save_dir, exist_ok=True)
        model.save_pretrained(best_model_save_dir)
        best_step = global_step

    if early_stopped:
        print(
            f"Early stopping: no Span_Strict_F1 gain for {patience} evals "
            f"(steps={global_step}, best={best_span_f1:.4f} @ step {best_step})"
        )
    avg_loss = total_loss / global_step if global_step else 0.0
    return {
        "avg_loss": avg_loss,
        "steps_trained": global_step,
        "best_span_strict_f1": best_span_f1,
        "best_step": best_step,
        "early_stopped": early_stopped,
        "num_evals": num_evals,
        "best_model_save_dir": best_model_save_dir,
    }


def get_labels(predictions, references, label_list: List[str]):
    true_predictions = []
    true_labels = []
    for pred_seq, ref_seq in zip(predictions, references):
        pred_labels = []
        ref_labels = []
        for pred_id, ref_id in zip(pred_seq, ref_seq):
            ref_id = ref_id.item()
            pred_id = pred_id.item()
            if ref_id != -100:
                pred_labels.append(label_list[pred_id])
                ref_labels.append(label_list[ref_id])
        true_predictions.append(pred_labels)
        true_labels.append(ref_labels)
    return true_predictions, true_labels


def evaluate_model(
    model,
    eval_dataloader,
    device,
    label_list,
    metric,
    *,
    show_progress: bool = False,
):
    model.eval()
    all_predictions = []
    all_labels = []
    batches = tqdm(eval_dataloader, desc="Eval", leave=False, disable=not show_progress)
    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        predicted_labels, true_labels = get_labels(predictions, labels, label_list)
        all_predictions.extend(predicted_labels)
        all_labels.extend(true_labels)

    results = metric.compute(predictions=all_predictions, references=all_labels)
    validation_metrics = {
        "Precision": results["overall_precision"],
        "Recall": results["overall_recall"],
        "F1": results["overall_f1"],
        "Accuracy": results["overall_accuracy"],
    }
    return validation_metrics, all_predictions, all_labels


def export_for_span_f1(dataset, predictions, true_labels, gold_file="gold.txt", pred_file="pred.txt"):
    with open(gold_file, "w", encoding="utf-8") as fg, open(pred_file, "w", encoding="utf-8") as fp:
        for i in range(len(dataset)):
            tokens = dataset[i][text_column_name]
            sentence_true_tags = true_labels[i]
            sentence_pred_tags = predictions[i]
            min_len = min(len(tokens), len(sentence_true_tags), len(sentence_pred_tags))
            tokens = tokens[:min_len]
            sentence_true_tags = sentence_true_tags[:min_len]
            sentence_pred_tags = sentence_pred_tags[:min_len]
            for token, true_tag, pred_tag in zip(tokens, sentence_true_tags, sentence_pred_tags):
                fg.write(f"{token}\t-\t{true_tag}\n")
                fp.write(f"{token}\t-\t{pred_tag}\n")
            fg.write("\n")
            fp.write("\n")


def span_scores_from_files(gold_file_path: str, pred_file_path: str) -> Dict[str, float]:
    gold_ners = span_f1.readNlu(gold_file_path)
    pred_ners = span_f1.readNlu(pred_file_path)

    tp = fp = fn = 0
    recall_loose_tp = recall_loose_fn = 0
    precision_loose_tp = precision_loose_fp = 0
    tp_ul = fp_ul = fn_ul = 0

    for gold_ner, pred_ner in zip(gold_ners, pred_ners):
        gold_spans = span_f1.toSpans(gold_ner)
        pred_spans = span_f1.toSpans(pred_ner)
        overlap = len(gold_spans.intersection(pred_spans))
        tp += overlap
        fp += len(pred_spans) - overlap
        fn += len(gold_spans) - overlap

        overlap_ul = span_f1.getUnlabeled(gold_spans, pred_spans)
        tp_ul += overlap_ul
        fp_ul += len(pred_spans) - overlap_ul
        fn_ul += len(gold_spans) - overlap_ul

        recall_loose_tp += span_f1.getLooseOverlap(gold_spans, pred_spans)
        recall_loose_fn += len(gold_spans) - span_f1.getLooseOverlap(gold_spans, pred_spans)

        precision_loose_tp += span_f1.getLooseOverlap(pred_spans, gold_spans)
        precision_loose_fp += len(pred_spans) - span_f1.getLooseOverlap(pred_spans, gold_spans)

    prec = 0.0 if tp + fp == 0 else tp / (tp + fp)
    rec = 0.0 if tp + fn == 0 else tp / (tp + fn)
    strict_f1 = 0.0 if prec + rec == 0.0 else 2 * (prec * rec) / (prec + rec)

    prec_ul = 0.0 if tp_ul + fp_ul == 0 else tp_ul / (tp_ul + fp_ul)
    rec_ul = 0.0 if tp_ul + fn_ul == 0 else tp_ul / (tp_ul + fn_ul)
    ul_f1 = 0.0 if prec_ul + rec_ul == 0.0 else 2 * (prec_ul * rec_ul) / (prec_ul + rec_ul)

    prec_l = (
        0.0
        if precision_loose_tp + precision_loose_fp == 0
        else precision_loose_tp / (precision_loose_tp + precision_loose_fp)
    )
    rec_l = (
        0.0
        if recall_loose_tp + recall_loose_fn == 0
        else recall_loose_tp / (recall_loose_tp + recall_loose_fn)
    )
    loose_f1 = 0.0 if prec_l + rec_l == 0.0 else 2 * (prec_l * rec_l) / (prec_l + rec_l)

    return {
        "Span_Strict_Precision": prec,
        "Span_Strict_Recall": rec,
        "Span_Strict_F1": strict_f1,
        "Span_Unlabeled_Precision": prec_ul,
        "Span_Unlabeled_Recall": rec_ul,
        "Span_Unlabeled_F1": ul_f1,
        "Span_Loose_Precision": prec_l,
        "Span_Loose_Recall": rec_l,
        "Span_Loose_F1": loose_f1,
    }


def build_row(
    *,
    k_shot: int,
    run_index: Optional[int],
    split_seed: Optional[int],
    row_type: str,
    n_train_sentences: int,
    max_train_steps_cap: int,
    train_steps_actual: int,
    best_span_f1_step: int,
    early_stopped: bool,
    es_eval_every: int,
    es_patience: int,
    es_min_steps_before_eval: int,
    validation_metrics: Dict[str, float],
    span_metrics: Dict[str, float],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "K_shot": k_shot,
        "Run_Index": run_index if run_index is not None else "",
        "Split_Seed": split_seed if split_seed is not None else "",
        "Row_Type": row_type,
        "N_Train_Sentences": n_train_sentences,
        "Max_Train_Steps": max_train_steps_cap,
        "Train_Steps": train_steps_actual,
        "Best_Span_F1_Step": best_span_f1_step,
        "Early_Stopped": int(early_stopped),
        "ES_Eval_Every": es_eval_every,
        "ES_Patience": es_patience,
        "ES_Min_Steps_Before_Eval": es_min_steps_before_eval,
        "Train_Batch_Size": train_batch_size,
        "Eval_Batch_Size": eval_batch_size,
        "Learning_Rate": learning_rate,
        "HF_Token_Precision": validation_metrics["Precision"],
        "HF_Token_Recall": validation_metrics["Recall"],
        "HF_Token_F1": validation_metrics["F1"],
        "HF_Token_Accuracy": validation_metrics["Accuracy"],
    }
    row.update(span_metrics)
    return row


CSV_FIELDNAMES = [
    "K_shot",
    "Run_Index",
    "Split_Seed",
    "Row_Type",
    "N_Train_Sentences",
    "Max_Train_Steps",
    "Train_Steps",
    "Best_Span_F1_Step",
    "Early_Stopped",
    "ES_Eval_Every",
    "ES_Patience",
    "ES_Min_Steps_Before_Eval",
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


def mean_numeric_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    skip = {
        "K_shot",
        "Run_Index",
        "Split_Seed",
        "Row_Type",
        "N_Train_Sentences",
        "Max_Train_Steps",
        "ES_Eval_Every",
        "ES_Patience",
        "ES_Min_Steps_Before_Eval",
    }
    keys = [k for k in CSV_FIELDNAMES if k not in skip]
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if r[k] != "" and r[k] is not None]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out


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

    ksplit = _load_k_split_module()
    raw_datasets = load_en_ewt_ud_iob2_datasets(_repo_root())
    label_list = list(raw_datasets["train"].features[label_column_name].feature.names)
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for i, label in enumerate(label_list)}

    n_train = len(raw_datasets["train"])
    examples_for_split: List[Mapping] = []
    for i in range(n_train):
        row = raw_datasets["train"][i]
        toks = row[text_column_name]
        labs = row[label_column_name]
        if labs and isinstance(labs[0], str):
            labs = [label_to_id[t] for t in labs]
        examples_for_split.append({"tokens": toks, "ner_tags": labs})

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    metric = evaluate.load("seqeval")

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=128,
            padding=False,
            truncation=True,
            is_split_into_words=True,
        )
        all_labels = []
        for batch_index, labels in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            label_ids = []
            prev_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id == prev_word_id:
                    label_ids.append(-100)
                else:
                    label_ids.append(labels[word_id])
                prev_word_id = word_id
            all_labels.append(label_ids)
        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reports_dir = os.path.join(_repo_root(), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    csv_file_path = os.path.join(reports_dir, results_filename)
    csv_file_exists = os.path.isfile(csv_file_path)
    csv_nonempty = csv_file_exists and os.path.getsize(csv_file_path) > 0
    csv_mode = "a" if csv_file_exists else "w"

    with open(csv_file_path, mode=csv_mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
        if not csv_nonempty:
            writer.writeheader()

        for k_shot in K_VALUES:
            run_rows: List[Dict[str, Any]] = []
            for run_index in range(NUM_RUNS_PER_K):
                split_seed = SPLIT_SEEDS[run_index]
                print(f"\n=== K={k_shot} run={run_index + 1}/{NUM_RUNS_PER_K} split_seed={split_seed} ===")

                set_seed(split_seed)

                support_list, _ = ksplit.greedy_k_shot_support_split(
                    examples_for_split,
                    k_shot,
                    labels_key=label_column_name,
                    id2label=id_to_label,
                    rng=random.Random(split_seed),
                )
                n_sup = len(support_list)
                print(f"Greedy K-shot support: {n_sup} sentences")

                train_hf = Dataset.from_dict(
                    {
                        "tokens": [x["tokens"] for x in support_list],
                        "ner_tags": [x["ner_tags"] for x in support_list],
                    },
                    features=raw_datasets["train"].features,
                )

                dd_run = DatasetDict(train=train_hf, validation=raw_datasets["validation"])
                processed = dd_run.map(
                    tokenize_and_align_labels,
                    batched=True,
                    remove_columns=dd_run["train"].column_names,
                    desc="Tokenize",
                )
                train_dataset = processed["train"]
                eval_dataset = processed["validation"]

                data_collator = DataCollatorForTokenClassification(tokenizer)
                train_dataloader = DataLoader(
                    train_dataset,
                    shuffle=True,
                    collate_fn=data_collator,
                    batch_size=train_batch_size,
                )
                eval_dataloader = DataLoader(
                    eval_dataset,
                    collate_fn=data_collator,
                    batch_size=eval_batch_size,
                )

                config = AutoConfig.from_pretrained(
                    model_name,
                    num_labels=len(label_list),
                    id2label=id_to_label,
                    label2id=label_to_id,
                )
                model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
                model.to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

                model_dir = os.path.join(
                    _repo_root(),
                    "models",
                    f"en_ewt_ud_fewshot_k{k_shot}_run{run_index}",
                )
                os.makedirs(model_dir, exist_ok=True)
                best_ckpt_dir = os.path.join(model_dir, "best_span_f1")

                es_summary = train_with_early_stopping(
                    model,
                    train_dataloader,
                    eval_dataloader,
                    optimizer,
                    device,
                    label_list,
                    metric,
                    max_steps=max_train_steps,
                    eval_every=early_stopping_eval_every,
                    patience=early_stopping_patience,
                    min_steps_before_eval=early_stopping_min_steps_before_eval,
                    best_model_save_dir=best_ckpt_dir,
                )
                print(
                    f"Training finished: steps={es_summary['steps_trained']}/{max_train_steps} "
                    f"avg_loss={es_summary['avg_loss']:.4f} "
                    f"best Span_Strict_F1={es_summary['best_span_strict_f1']:.4f} "
                    f"at step {es_summary['best_step']} "
                    f"early_stopped={es_summary['early_stopped']}"
                )

                # Reload best checkpoint (span F1) for final metrics and exported artifacts.
                model = AutoModelForTokenClassification.from_pretrained(best_ckpt_dir)
                model.to(device)

                validation_metrics, all_predictions, all_labels = evaluate_model(
                    model,
                    eval_dataloader,
                    device,
                    label_list,
                    metric,
                    show_progress=False,
                )
                span_metrics = span_metrics_from_aligned_word_tags(all_labels, all_predictions)

                out_dir = os.path.join(
                    _repo_root(),
                    "data",
                    "processed",
                    f"en_ewt_ud_fewshot_k{k_shot}_run{run_index}",
                )
                os.makedirs(out_dir, exist_ok=True)
                gold_fp = os.path.join(out_dir, "gold.txt")
                pred_fp = os.path.join(out_dir, "pred.txt")
                export_for_span_f1(
                    raw_datasets["validation"],
                    all_predictions,
                    all_labels,
                    gold_file=gold_fp,
                    pred_file=pred_fp,
                )
                print(
                    f"Best-checkpoint eval — Span strict F1: {span_metrics['Span_Strict_F1']:.4f} | "
                    f"HF token F1: {validation_metrics['F1']:.4f}"
                )

                model.save_pretrained(model_dir)
                tokenizer.save_pretrained(model_dir)

                row = build_row(
                    k_shot=k_shot,
                    run_index=run_index,
                    split_seed=split_seed,
                    row_type="run",
                    n_train_sentences=n_sup,
                    max_train_steps_cap=max_train_steps,
                    train_steps_actual=es_summary["steps_trained"],
                    best_span_f1_step=es_summary["best_step"],
                    early_stopped=es_summary["early_stopped"],
                    es_eval_every=early_stopping_eval_every,
                    es_patience=early_stopping_patience,
                    es_min_steps_before_eval=early_stopping_min_steps_before_eval,
                    validation_metrics=validation_metrics,
                    span_metrics=span_metrics,
                )
                writer.writerow({k: row.get(k, "") for k in CSV_FIELDNAMES})
                csvfile.flush()
                run_rows.append(row)

            means = mean_numeric_rows(run_rows)
            mean_row: Dict[str, Any] = {
                "K_shot": k_shot,
                "Run_Index": "",
                "Split_Seed": "",
                "Row_Type": "mean_over_runs",
                "N_Train_Sentences": int(
                    sum(r["N_Train_Sentences"] for r in run_rows) / len(run_rows)
                ),
                "Max_Train_Steps": max_train_steps,
                "Train_Batch_Size": train_batch_size,
                "Eval_Batch_Size": eval_batch_size,
                "Learning_Rate": learning_rate,
                "ES_Eval_Every": early_stopping_eval_every,
                "ES_Patience": early_stopping_patience,
                "ES_Min_Steps_Before_Eval": early_stopping_min_steps_before_eval,
            }
            for k in CSV_FIELDNAMES:
                if k in mean_row:
                    continue
                mean_row[k] = means.get(k, "")
            writer.writerow({k: mean_row.get(k, "") for k in CSV_FIELDNAMES})
            csvfile.flush()
            print(
                f"\n--- K={k_shot} mean over {NUM_RUNS_PER_K} runs: "
                f"HF F1={mean_row['HF_Token_F1']:.4f} "
                f"Span strict F1={mean_row['Span_Strict_F1']:.4f} ---"
            )

    action = "appended to" if csv_nonempty else "written to"
    print(f"\nAll runs and per-K means {action} {csv_file_path}")


if __name__ == "__main__":
    main()
