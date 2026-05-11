from __future__ import annotations

from typing import Dict, List, Tuple

import evaluate
import torch
import utils.span_f1
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorForTokenClassification


TEXT_COLUMN = "tokens"
LABEL_COLUMN = "ner_tags"


def tokenize_and_align_labels_fn(tokenizer, label_to_id):
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[TEXT_COLUMN],
            max_length=128,
            padding=False,
            truncation=True,
            is_split_into_words=True,
        )

        all_labels = []

        for batch_index, labels in enumerate(examples[LABEL_COLUMN]):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            label_ids = []
            prev_word_id = None

            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id == prev_word_id:
                    label_ids.append(-100)
                else:
                    label = labels[word_id]
                    if isinstance(label, str):
                        label = label_to_id[label]
                    label_ids.append(label)

                prev_word_id = word_id

            all_labels.append(label_ids)

        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    return tokenize_and_align_labels


def make_dataloader(dataset, tokenizer, batch_size: int, shuffle: bool = False):
    data_collator = DataCollatorForTokenClassification(tokenizer)

    return DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=data_collator,
        batch_size=batch_size,
    )


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


def span_metrics_from_aligned_word_tags(
    gold_tags_per_sent: List[List[str]],
    pred_tags_per_sent: List[List[str]],
) -> Dict[str, float]:
    tp = fp = fn = 0
    recall_loose_tp = recall_loose_fn = 0
    precision_loose_tp = precision_loose_fp = 0
    tp_ul = fp_ul = fn_ul = 0

    for gold_ner, pred_ner in zip(gold_tags_per_sent, pred_tags_per_sent):
        gold_spans = utils.span_f1.toSpans(gold_ner)
        pred_spans = utils.span_f1.toSpans(pred_ner)

        overlap = len(gold_spans.intersection(pred_spans))
        tp += overlap
        fp += len(pred_spans) - overlap
        fn += len(gold_spans) - overlap

        overlap_ul = utils.span_f1.getUnlabeled(gold_spans, pred_spans)
        tp_ul += overlap_ul
        fp_ul += len(pred_spans) - overlap_ul
        fn_ul += len(gold_spans) - overlap_ul

        recall_loose = utils.span_f1.getLooseOverlap(gold_spans, pred_spans)
        precision_loose = utils.span_f1.getLooseOverlap(pred_spans, gold_spans)

        recall_loose_tp += recall_loose
        recall_loose_fn += len(gold_spans) - recall_loose

        precision_loose_tp += precision_loose
        precision_loose_fp += len(pred_spans) - precision_loose

    prec = 0.0 if tp + fp == 0 else tp / (tp + fp)
    rec = 0.0 if tp + fn == 0 else tp / (tp + fn)
    strict_f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)

    prec_ul = 0.0 if tp_ul + fp_ul == 0 else tp_ul / (tp_ul + fp_ul)
    rec_ul = 0.0 if tp_ul + fn_ul == 0 else tp_ul / (tp_ul + fn_ul)
    ul_f1 = 0.0 if prec_ul + rec_ul == 0 else 2 * prec_ul * rec_ul / (prec_ul + rec_ul)

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
    loose_f1 = 0.0 if prec_l + rec_l == 0 else 2 * prec_l * rec_l / (prec_l + rec_l)

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


def evaluate_model(
    model,
    eval_dataloader,
    device,
    label_list: List[str],
    *,
    show_progress: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    metric = evaluate.load("seqeval")

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

    results = metric.compute(
        predictions=all_predictions,
        references=all_labels,
    )

    token_metrics = {
        "HF_Token_Precision": results["overall_precision"],
        "HF_Token_Recall": results["overall_recall"],
        "HF_Token_F1": results["overall_f1"],
        "HF_Token_Accuracy": results["overall_accuracy"],
    }

    span_metrics = span_metrics_from_aligned_word_tags(
        all_labels,
        all_predictions,
    )

    return token_metrics, span_metrics