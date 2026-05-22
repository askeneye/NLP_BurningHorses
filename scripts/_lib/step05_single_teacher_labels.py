from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from scripts._lib.repro_io import load_yaml, read_json_lines, write_json, write_jsonl
from scripts._lib.step04_predict_bart import BRACKETED_ENTITY_RE, find_entity_span, normalize_entity_type
from scripts._lib.teacher_data import apply_pattern, detokenize_tokens


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def resolve_single_teacher_label_paths(root: Path, config: dict[str, Any]) -> dict[str, Path | str]:
    split_name = str(config.get("default_profile", {}).get("split_name", "k5_seed242"))
    paths = config.get("paths", {})
    interim_root = root / str(paths.get("interim_root", "reproduction/data/interim"))
    logs_root = root / str(paths.get("logs_root", "reproduction/logs"))
    pattern_file = root / str(paths.get("pattern_bank_file", "scripts/assets/patterns.yaml"))
    return {
        "split_name": split_name,
        "base_root": interim_root / split_name / "base",
        "teacher_predictions_root": interim_root / split_name / "teacher_predictions",
        "output_root": interim_root / split_name / "single_teacher_labels",
        "step04_manifest": logs_root / "step04_predict_bart_manifest.json",
        "step05_manifest": logs_root / "step05_single_teacher_labels_manifest.json",
        "pattern_file": pattern_file,
    }


def _entity_types_from_metadata(metadata_path: Path) -> set[str]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {str(label)[2:] for label in metadata["label_list"] if str(label).startswith("B-")}


def _parse_validity(row: dict[str, Any], entity_types: set[str]) -> dict[str, Any]:
    tokens = [str(token) for token in row["tokens"]]
    generated_text = str(row.get("generated_text", ""))
    occupied_indices: set[int] = set()
    matched_spans = 0
    invalid_spans = 0

    for match in BRACKETED_ENTITY_RE.finditer(generated_text):
        entity_text = match.group(1).strip()
        entity_type = normalize_entity_type(match.group(2).strip())
        if entity_type not in entity_types:
            invalid_spans += 1
            continue
        span = find_entity_span(tokens, entity_text, occupied_indices)
        if span is None:
            invalid_spans += 1
            continue
        start_index, end_index = span
        occupied_indices.update(range(start_index, end_index))
        matched_spans += 1

    predicted_tags = [str(tag) for tag in row["predicted_tags"]]
    predicted_entity_tokens = sum(1 for tag in predicted_tags if tag != "O")
    has_entity = predicted_entity_tokens > 0
    return {
        "parse_valid": invalid_spans == 0 and (matched_spans > 0 or not has_entity),
        "matched_generated_spans": matched_spans,
        "invalid_generated_spans": invalid_spans,
        "has_entity": has_entity,
        "entity_token_count": predicted_entity_tokens,
    }


def _sequence_confidences(
    *,
    rows: list[dict[str, Any]],
    checkpoint_dir: Path,
    pattern_template: str,
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
) -> list[dict[str, float | int]]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
    tokenizer.model_max_length = max_source_length
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    scores: list[dict[str, float | int]] = []
    for start in tqdm(
        range(0, len(rows), batch_size),
        desc="single_teacher_confidence",
        dynamic_ncols=True,
        mininterval=5.0,
    ):
        batch = rows[start : start + batch_size]
        input_texts = [
            apply_pattern(
                sentence_text=detokenize_tokens(row["tokens"]),
                order_text="first to last",
                pattern_template=pattern_template,
            )
            for row in batch
        ]
        target_texts = [str(row.get("generated_text", "")).strip() for row in batch]
        tokenized_inputs = tokenizer(
            input_texts,
            max_length=max_source_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        tokenized_targets = tokenizer(
            text_target=target_texts,
            max_length=max_target_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels = tokenized_targets["input_ids"].to(device)
        label_mask = labels != tokenizer.pad_token_id
        labels_for_model = labels.masked_fill(~label_mask, -100)

        with torch.no_grad():
            outputs = model(**tokenized_inputs, labels=labels_for_model)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)
            gather_labels = labels.masked_fill(~label_mask, 0).unsqueeze(-1)
            token_log_probs = log_probs.gather(-1, gather_labels).squeeze(-1)
            token_log_probs = token_log_probs.masked_fill(~label_mask, 0.0)
            lengths = label_mask.sum(dim=1).clamp_min(1)
            sums = token_log_probs.sum(dim=1)
            means = sums / lengths
            mins = token_log_probs.masked_fill(~label_mask, float("inf")).min(dim=1).values

        for index in range(len(batch)):
            scores.append(
                {
                    "avg_token_logprob": float(means[index].detach().cpu()),
                    "sum_token_logprob": float(sums[index].detach().cpu()),
                    "min_token_logprob": float(mins[index].detach().cpu()),
                    "target_token_count": int(lengths[index].detach().cpu()),
                }
            )
    return scores


def _select_rows(
    *,
    rows: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
    subset_size: int,
    max_no_entity_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_scored = [(index, score) for index, score in enumerate(scored_rows) if score["parse_valid"]]
    entity_rows = [item for item in valid_scored if item[1]["has_entity"]]
    no_entity_rows = [item for item in valid_scored if not item[1]["has_entity"]]
    ranked_entity_rows = sorted(
        entity_rows,
        key=lambda item: (
            item[1]["avg_token_logprob"],
            item[1]["min_token_logprob"],
            item[1]["entity_token_count"],
        ),
        reverse=True,
    )
    ranked_no_entity_rows = sorted(
        no_entity_rows,
        key=lambda item: (item[1]["avg_token_logprob"], item[1]["min_token_logprob"]),
        reverse=True,
    )

    max_no_entity = int(subset_size * max_no_entity_fraction)
    selected = ranked_entity_rows[: subset_size - max_no_entity]
    selected.extend(ranked_no_entity_rows[:max_no_entity])
    if len(selected) < subset_size:
        selected_indices = {index for index, _ in selected}
        backfill = [item for item in ranked_entity_rows if item[0] not in selected_indices]
        selected.extend(backfill[: subset_size - len(selected)])
    if len(selected) < subset_size:
        selected_indices = {index for index, _ in selected}
        backfill = [item for item in ranked_no_entity_rows if item[0] not in selected_indices]
        selected.extend(backfill[: subset_size - len(selected)])

    selected_indices = sorted(index for index, _ in selected[:subset_size])
    selected_scores = [scored_rows[index] for index in selected_indices]
    selected_rows: list[dict[str, Any]] = []
    for index in selected_indices:
        row = dict(rows[index])
        row["teacher"] = "hard_argmax"
        row["self_confidence"] = {
            key: scored_rows[index][key]
            for key in [
                "avg_token_logprob",
                "sum_token_logprob",
                "min_token_logprob",
                "target_token_count",
                "parse_valid",
                "matched_generated_spans",
                "invalid_generated_spans",
            ]
        }
        selected_rows.append(row)

    stats = {
        "requested_subset_size": subset_size,
        "selected_rows": len(selected_rows),
        "valid_rows": len(valid_scored),
        "valid_with_entities": len(entity_rows),
        "valid_without_entities": len(no_entity_rows),
        "selected_with_entities": sum(1 for score in selected_scores if score["has_entity"]),
        "selected_without_entities": sum(1 for score in selected_scores if not score["has_entity"]),
        "mean_avg_token_logprob": (
            sum(float(score["avg_token_logprob"]) for score in selected_scores) / max(1, len(selected_scores))
        ),
        "mean_min_token_logprob": (
            sum(float(score["min_token_logprob"]) for score in selected_scores) / max(1, len(selected_scores))
        ),
        "selection_policy": (
            "Teacher-force score the generated single-teacher outputs. Keep parse-valid rows, rank entity-containing "
            "rows by average target-token log probability, reserve up to the configured fraction for confident "
            "no-entity rows, then restore source order."
        ),
    }
    return selected_rows, stats


def build_single_teacher_labels(
    *,
    root: Path,
    resolved_paths: dict[str, Path | str],
    config: dict[str, Any],
    pattern_limit: int | None,
) -> dict[str, Any]:
    started_at = time.time()
    teacher_outputs_cfg = config.get("teacher_outputs", {})
    subset_size = int(teacher_outputs_cfg.get("filter_top_k", 1000))
    max_no_entity_fraction = float(teacher_outputs_cfg.get("max_no_entity_fraction", 0.1))
    split_name = str(resolved_paths["split_name"])
    step04_manifest_path = Path(resolved_paths["step04_manifest"])
    if not step04_manifest_path.exists():
        raise FileNotFoundError(f"Missing Step-04 manifest: {step04_manifest_path}")
    step04_manifest = json.loads(step04_manifest_path.read_text(encoding="utf-8"))
    jobs = [job for job in step04_manifest.get("jobs", []) if str(job.get("split_name")) == split_name]
    if pattern_limit is not None:
        jobs = jobs[:pattern_limit]
    if len(jobs) != 1:
        raise ValueError("Single-teacher label preparation expects exactly one Step-04 job.")
    job = jobs[0]
    source_pattern_id = str(job["source_pattern_id"])
    pattern_records = load_yaml(Path(resolved_paths["pattern_file"])).get("patterns", [])
    source_template = next(
        (str(pattern["template"]) for pattern in pattern_records if str(pattern.get("id")) == source_pattern_id),
        None,
    )
    if source_template is None:
        raise ValueError(f"Missing source pattern template for {source_pattern_id}")

    prediction_file = Path(job["output_root"]) / "unlabeled_pool" / "hard_predictions.jsonl"
    checkpoint_dir = Path(job["checkpoint_dir"])
    rows = read_json_lines(prediction_file)
    entity_types = _entity_types_from_metadata(Path(resolved_paths["base_root"]) / "metadata.json")
    scores = _sequence_confidences(
        rows=rows,
        checkpoint_dir=checkpoint_dir,
        pattern_template=source_template,
        batch_size=16,
        max_source_length=128,
        max_target_length=128,
    )
    scored_rows = [{**score, **_parse_validity(row, entity_types)} for row, score in zip(rows, scores)]
    selected_rows, stats = _select_rows(
        rows=rows,
        scored_rows=scored_rows,
        subset_size=subset_size,
        max_no_entity_fraction=max_no_entity_fraction,
    )

    output_dir = Path(resolved_paths["output_root"]) / "unlabeled_pool" / f"top_{subset_size}"
    output_file = output_dir / "hard_labels.jsonl"
    manifest_file = output_dir / "manifest.json"
    write_jsonl(output_file, selected_rows)
    manifest = {
        "dataset": "conll2003",
        "source_method": "single_teacher_self_confidence",
        "kshot_split": split_name,
        "eval_split": "unlabeled_pool",
        "subset_name": f"top_{subset_size}",
        "source_prediction_file": _relative(root, prediction_file),
        "source_checkpoint_dir": _relative(root, checkpoint_dir),
        "source_pattern": str(job["pattern_id"]),
        "source_pattern_id": source_pattern_id,
        "source_template": source_template,
        "outputs": {"hard_argmax": _relative(root, output_file)},
        **stats,
        "elapsed_seconds": time.time() - started_at,
    }
    write_json(manifest_file, manifest)
    write_json(Path(resolved_paths["step05_manifest"]), {"label_file": str(output_file), **manifest})
    print(
        f"Prepared single-teacher labels selected={manifest['selected_rows']} "
        f"valid={manifest['valid_rows']} mean_logp={manifest['mean_avg_token_logprob']:.3f}"
    )
    print(f"label_file={output_file}")
    print(f"run_manifest={resolved_paths['step05_manifest']}")
    return {"label_file": str(output_file), **manifest}
