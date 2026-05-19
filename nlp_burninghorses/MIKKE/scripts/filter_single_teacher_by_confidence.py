from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start_path: Path) -> Path:
    for path in [start_path, *start_path.parents]:
        if (path / "pyproject.toml").exists():
            return path
    raise FileNotFoundError("Could not locate project root containing pyproject.toml.")


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __package__:
    from .bart import (
        BRACKETED_ENTITY_RE,
        detokenize_tokens,
        find_entity_span,
        normalize_entity_type,
    )
    from .data_aug_train import repo_relative_path
    from .generate_bart_member_soft_labels import (
        apply_member_wrapper,
        read_json,
        read_jsonl,
        select_pattern,
        source_pattern_from_training_summary,
    )
else:
    from bart import (
        BRACKETED_ENTITY_RE,
        detokenize_tokens,
        find_entity_span,
        normalize_entity_type,
    )
    from data_aug_train import repo_relative_path
    from generate_bart_member_soft_labels import (
        apply_member_wrapper,
        read_json,
        read_jsonl,
        select_pattern,
        source_pattern_from_training_summary,
    )

import yaml


DEFAULT_SPLITS = [
    "k5_seed42",
    "k5_seed142",
    "k5_seed242",
    "k10_seed42",
    "k10_seed142",
    "k10_seed242",
    "k20_seed42",
    "k20_seed142",
    "k20_seed242",
    "k50_seed42",
    "k50_seed142",
    "k50_seed242",
]
DEFAULT_MODEL_ROOT = "models/conll2003_kshot_seq2seq/bart_base/pet_oada_verbalized"
DEFAULT_PREDICTION_ROOT = "data/interim/conll2003_soft_labels/bart_base/pet_oada_verbalized"
DEFAULT_OUTPUT_ROOT = (
    "data/interim/conll2003_single_teacher_teachers/"
    "bart_base/pet_oada_verbalized_final2000_self_confidence"
)
DEFAULT_METADATA_FILE = "data/interim/conll2003_kshot_bert/metadata.json"
DEFAULT_PATTERNS_PATH = "nlp_burninghorses/MIKKE/scripts/patterns.yaml"
DEFAULT_PATTERN_SECTION = "patterns"
DEFAULT_PATTERN = "pattern_01"
DEFAULT_SUFFIX = "xe_verbalized_final2000"
DEFAULT_EVAL_SPLIT = "unlabeled_pool"
DEFAULT_CHECKPOINT_SUBDIR = "final_step"
DEFAULT_INFERENCE_ORDER = "first to last"
DEFAULT_SUBSET_SIZE = 1000
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_SOURCE_LENGTH = 128
DEFAULT_MAX_TARGET_LENGTH = 128
DEFAULT_NO_ENTITY_FRACTION = 0.1


@dataclass(frozen=True)
class ConfidenceConfig:
    splits: list[str]
    model_root: Path
    prediction_root: Path
    output_root: Path
    metadata_file: Path
    patterns_path: Path
    pattern_section: str
    pattern: str
    suffix: str
    eval_split: str
    checkpoint_subdir: str
    inference_order: str
    subset_size: int
    batch_size: int
    max_source_length: int
    max_target_length: int
    no_entity_fraction: float


def csv_from_env(name: str, default: list[str]) -> list[str]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return yaml.safe_load(input_file) or {}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)


def label_list_from_metadata(metadata_path: Path) -> list[str]:
    metadata = read_json(metadata_path)
    return [str(label) for label in metadata["label_list"]]


def entity_types_from_label_list(label_list: list[str]) -> set[str]:
    return {label.split("-", 1)[1] for label in label_list if "-" in label}


def model_dir_for(config: ConfidenceConfig, split: str) -> Path:
    return config.model_root / split / f"{config.pattern}_{config.suffix}"


def prediction_file_for(config: ConfidenceConfig, split: str) -> Path:
    return (
        config.prediction_root
        / split
        / f"{config.pattern}_{config.suffix}"
        / config.eval_split
        / "hard_predictions.jsonl"
    )


def output_dir_for(config: ConfidenceConfig, split: str) -> Path:
    return config.output_root / split / config.eval_split / f"top_{config.subset_size}"


def parse_validity(row: dict[str, Any], entity_types: set[str]) -> dict[str, Any]:
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
    valid = invalid_spans == 0 and (matched_spans > 0 or not has_entity)
    return {
        "parse_valid": valid,
        "matched_generated_spans": matched_spans,
        "invalid_generated_spans": invalid_spans,
        "has_entity": has_entity,
        "entity_token_count": predicted_entity_tokens,
    }


def sequence_confidences(
    rows: list[dict[str, Any]],
    config: ConfidenceConfig,
    model_dir: Path,
    pattern_template: str,
) -> list[dict[str, float]]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir / config.checkpoint_subdir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir / config.checkpoint_subdir)
    tokenizer.model_max_length = config.max_source_length
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    scores: list[dict[str, float]] = []
    for batch_start in tqdm(
        range(0, len(rows), config.batch_size),
        desc=f"Score {model_dir.parent.name}",
        dynamic_ncols=True,
    ):
        batch = rows[batch_start : batch_start + config.batch_size]
        input_texts = [
            apply_member_wrapper(
                detokenize_tokens(row["tokens"]),
                pattern_template,
                config.inference_order,
            )
            for row in batch
        ]
        target_texts = [str(row.get("generated_text", "")).strip() for row in batch]
        tokenized_inputs = tokenizer(
            input_texts,
            max_length=config.max_source_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        tokenized_targets = tokenizer(
            text_target=target_texts,
            max_length=config.max_target_length,
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


def select_rows(
    rows: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
    config: ConfidenceConfig,
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

    max_no_entity = int(config.subset_size * config.no_entity_fraction)
    entity_quota = config.subset_size - max_no_entity
    selected = ranked_entity_rows[:entity_quota]
    selected.extend(ranked_no_entity_rows[:max_no_entity])
    if len(selected) < config.subset_size:
        selected_indices_so_far = {index for index, _ in selected}
        backfill = [
            item
            for item in ranked_entity_rows[entity_quota:]
            if item[0] not in selected_indices_so_far
        ]
        selected.extend(backfill[: config.subset_size - len(selected)])
    if len(selected) < config.subset_size:
        selected_indices_so_far = {index for index, _ in selected}
        backfill = [
            item
            for item in ranked_no_entity_rows[max_no_entity:]
            if item[0] not in selected_indices_so_far
        ]
        selected.extend(backfill[: config.subset_size - len(selected)])

    selected = selected[: config.subset_size]
    selected_indices = sorted(index for index, _ in selected)
    selected_scores = [scored_rows[index] for index in selected_indices]
    selected_rows = []
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

    manifest_stats = {
        "requested_subset_size": config.subset_size,
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
            "Teacher-force score the already generated single-teacher output. Keep parse-valid rows, "
            "rank entity-containing rows by average target-token log probability, reserve up to "
            f"{config.no_entity_fraction:.0%} for confident no-entity rows, then restore source order."
        ),
    }
    return selected_rows, manifest_stats


def filter_split(config: ConfidenceConfig, split: str, label_list: list[str]) -> dict[str, Any]:
    started_at = time.time()
    model_dir = model_dir_for(config, split)
    prediction_file = prediction_file_for(config, split)
    output_dir = output_dir_for(config, split)
    source_pattern_id = source_pattern_from_training_summary(model_dir)
    if source_pattern_id is None:
        raise ValueError(f"Could not determine source pattern id from {model_dir}")
    pattern = select_pattern(load_yaml(config.patterns_path), config.pattern_section, source_pattern_id)
    rows = read_jsonl(prediction_file)
    entity_types = entity_types_from_label_list(label_list)

    confidence_scores = sequence_confidences(rows, config, model_dir, pattern["template"])
    scored_rows = []
    for row, confidence in zip(rows, confidence_scores):
        scored_rows.append({**confidence, **parse_validity(row, entity_types)})

    selected_rows, stats = select_rows(rows, scored_rows, config)
    output_file = output_dir / "ensemble_hard_argmax.jsonl"
    manifest_file = output_dir / "manifest.json"
    write_jsonl(output_file, selected_rows)
    manifest = {
        "dataset": "conll2003",
        "source_method": "pet_oada_verbalized_single_teacher",
        "source_suffix": config.suffix,
        "kshot_split": split,
        "eval_split": config.eval_split,
        "subset_name": f"top_{config.subset_size}",
        "source_prediction_file": repo_relative_path(prediction_file),
        "source_model_dir": repo_relative_path(model_dir),
        "source_pattern": config.pattern,
        "source_pattern_id": source_pattern_id,
        "source_template": pattern["template"],
        "inference_order": config.inference_order,
        "outputs": {"hard_argmax": repo_relative_path(output_file)},
        **stats,
        "elapsed_seconds": time.time() - started_at,
    }
    write_json(manifest_file, manifest)
    print(
        f"{split}: selected={manifest['selected_rows']} valid={manifest['valid_rows']} "
        f"entities={manifest['selected_with_entities']} mean_logp={manifest['mean_avg_token_logprob']:.3f}"
    )
    return manifest


def load_config_from_env() -> ConfidenceConfig:
    return ConfidenceConfig(
        splits=csv_from_env("SINGLE_TEACHER_SPLITS", DEFAULT_SPLITS),
        model_root=(PROJECT_ROOT / os.environ.get("SINGLE_TEACHER_MODEL_ROOT", DEFAULT_MODEL_ROOT)).resolve(),
        prediction_root=(
            PROJECT_ROOT / os.environ.get("SINGLE_TEACHER_PREDICTION_ROOT", DEFAULT_PREDICTION_ROOT)
        ).resolve(),
        output_root=(PROJECT_ROOT / os.environ.get("SINGLE_TEACHER_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)).resolve(),
        metadata_file=(PROJECT_ROOT / os.environ.get("SINGLE_TEACHER_METADATA_FILE", DEFAULT_METADATA_FILE)).resolve(),
        patterns_path=(PROJECT_ROOT / os.environ.get("SINGLE_TEACHER_PATTERNS_PATH", DEFAULT_PATTERNS_PATH)).resolve(),
        pattern_section=os.environ.get("SINGLE_TEACHER_PATTERN_SECTION", DEFAULT_PATTERN_SECTION),
        pattern=os.environ.get("SINGLE_TEACHER_PATTERN", DEFAULT_PATTERN),
        suffix=os.environ.get("SINGLE_TEACHER_SUFFIX", DEFAULT_SUFFIX),
        eval_split=os.environ.get("SINGLE_TEACHER_EVAL_SPLIT", DEFAULT_EVAL_SPLIT),
        checkpoint_subdir=os.environ.get("SINGLE_TEACHER_CHECKPOINT_SUBDIR", DEFAULT_CHECKPOINT_SUBDIR),
        inference_order=os.environ.get("SINGLE_TEACHER_INFERENCE_ORDER", DEFAULT_INFERENCE_ORDER),
        subset_size=int(os.environ.get("SINGLE_TEACHER_SUBSET_SIZE", DEFAULT_SUBSET_SIZE)),
        batch_size=int(os.environ.get("SINGLE_TEACHER_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        max_source_length=int(os.environ.get("SINGLE_TEACHER_MAX_SOURCE_LENGTH", DEFAULT_MAX_SOURCE_LENGTH)),
        max_target_length=int(os.environ.get("SINGLE_TEACHER_MAX_TARGET_LENGTH", DEFAULT_MAX_TARGET_LENGTH)),
        no_entity_fraction=float(os.environ.get("SINGLE_TEACHER_NO_ENTITY_FRACTION", DEFAULT_NO_ENTITY_FRACTION)),
    )


def main() -> None:
    config = load_config_from_env()
    label_list = label_list_from_metadata(config.metadata_file)
    print("Single-teacher confidence filtering")
    print(f"splits={', '.join(config.splits)}")
    print(f"subset_size={config.subset_size}")
    for split in config.splits:
        filter_split(config, split, label_list)


if __name__ == "__main__":
    main()
