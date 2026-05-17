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

from nlp_burninghorses.MIKKE.scripts.bart import (  # noqa: E402
    BRACKETED_ENTITY_RE,
    detokenize_tokens,
    find_entity_span,
    generated_text_to_bio_tags,
    normalize_entity_type,
    read_jsonl,
)
from nlp_burninghorses.MIKKE.scripts.generate_bart_member_soft_labels import (  # noqa: E402
    apply_member_wrapper,
    generation_step_offsets,
    generated_ids_for_scores,
    gold_tags_from_row,
    load_yaml,
    select_pattern,
    source_pattern_from_training_summary,
)
from nlp_burninghorses.MIKKE.scripts.probe_bart_verbalizer_logits import (  # noqa: E402
    aligned_score_step,
    candidate_probabilities,
    parse_label_verbalizer,
    verbalizer_token_ids,
)

DEFAULT_INPUT_FILE = "data/interim/conll2003_kshot_bert/k5_seed42/unlabeled_pool.jsonl"
DEFAULT_METADATA_FILE = "data/interim/conll2003_kshot_bert/metadata.json"
DEFAULT_MODEL_DIR = (
    "models/conll2003_kshot_seq2seq/bart_base/pet_oada_verbalized/"
    "k5_seed42/pattern_01_xe_verbalized_pilot1000"
)
DEFAULT_CHECKPOINT_SUBDIR = "final_step"
DEFAULT_OUTPUT_DIR = (
    "data/interim/conll2003_true_soft_labels/bart_base/pet_oada_verbalized/"
    "k5_seed42/pattern_01_xe_verbalized_pilot1000/unlabeled_pool"
)
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_SOURCE_LENGTH = 128
DEFAULT_MAX_GENERATION_LENGTH = 128
DEFAULT_PATTERNS_PATH = "nlp_burninghorses/MIKKE/scripts/patterns.yaml"
DEFAULT_PATTERN_SECTION = "patterns"
DEFAULT_INFERENCE_ORDER = "first to last"


@dataclass(frozen=True)
class TrueSoftConfig:
    input_file: Path
    metadata_file: Path
    model_dir: Path
    checkpoint_dir: Path
    output_dir: Path
    soft_label_file: Path
    hard_prediction_file: Path
    patterns_path: Path
    pattern_section: str
    source_pattern_id: str | None
    inference_order: str
    label_verbalizer: dict[str, str]
    batch_size: int
    max_source_length: int
    max_generation_length: int


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)


def repo_relative_path(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved_path)


def label_list_from_metadata(metadata_path: Path) -> list[str]:
    label_list = read_json(metadata_path).get("label_list")
    if not isinstance(label_list, list) or not label_list:
        raise ValueError(f"{metadata_path} must contain a non-empty label_list.")
    return [str(label) for label in label_list]


def entity_types_from_label_list(label_list: list[str]) -> set[str]:
    return {label.split("-", 1)[1] for label in label_list if "-" in label}


def o_distribution(label_list: list[str]) -> list[float]:
    distribution = [0.0] * len(label_list)
    distribution[label_list.index("O")] = 1.0
    return distribution


def bio_distribution(
    prefix: str,
    type_probabilities: dict[str, float],
    label_list: list[str],
) -> list[float]:
    distribution = [0.0] * len(label_list)
    for entity_type, probability in type_probabilities.items():
        label = f"{prefix}-{entity_type}"
        if label in label_list:
            distribution[label_list.index(label)] = probability
    total = sum(distribution)
    if total <= 0:
        return o_distribution(label_list)
    return [value / total for value in distribution]


def generated_true_soft_labels(
    tokenizer,
    tokens: list[str],
    generated_text: str,
    generated_ids: list[int],
    score_tensors: list[torch.Tensor],
    batch_index: int,
    candidate_token_ids: dict[str, list[list[int]]],
    label_list: list[str],
    entity_types: set[str],
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    soft_labels = [o_distribution(label_list) for _ in tokens]
    occupied_indices: set[int] = set()
    step_offsets = generation_step_offsets(tokenizer, generated_ids)
    entity_records: list[dict[str, Any]] = []

    for match in BRACKETED_ENTITY_RE.finditer(generated_text):
        entity_text = match.group(1).strip()
        canonical_label = normalize_entity_type(match.group(2).strip())
        if canonical_label not in entity_types:
            continue
        span = find_entity_span(tokens, entity_text, occupied_indices)
        if span is None:
            continue
        decoded_step = None
        span_start, span_end = match.span(2)
        for step_index, (step_start, step_end) in enumerate(step_offsets):
            if step_end > span_start and step_start < span_end:
                decoded_step = step_index
                break
        if decoded_step is None:
            continue
        generated_token_id = generated_ids[decoded_step] if decoded_step < len(generated_ids) else None
        score_step = aligned_score_step(
            score_tensors,
            batch_index,
            generated_token_id,
            decoded_step,
            candidate_token_ids,
        )
        type_probabilities = candidate_probabilities(score_tensors[score_step][batch_index], candidate_token_ids)
        start_index, end_index = span
        for token_index in range(start_index, end_index):
            prefix = "B" if token_index == start_index else "I"
            soft_labels[token_index] = bio_distribution(prefix, type_probabilities, label_list)
        occupied_indices.update(range(start_index, end_index))
        entity_records.append(
            {
                "entity_text": entity_text,
                "surface_label": match.group(2).strip(),
                "canonical_label": canonical_label,
                "span": [start_index, end_index],
                "decoded_type_step": decoded_step,
                "score_step": score_step,
                "type_probabilities": type_probabilities,
            }
        )
    return soft_labels, entity_records


def load_config_from_env() -> TrueSoftConfig:
    model_dir = Path(os.environ.get("TRUE_SOFT_MODEL_DIR", PROJECT_ROOT / DEFAULT_MODEL_DIR)).resolve()
    output_dir = Path(os.environ.get("TRUE_SOFT_OUTPUT_DIR", PROJECT_ROOT / DEFAULT_OUTPUT_DIR)).resolve()
    checkpoint_dir = Path(
        os.environ.get("TRUE_SOFT_CHECKPOINT_DIR", model_dir / DEFAULT_CHECKPOINT_SUBDIR)
    ).resolve()
    return TrueSoftConfig(
        input_file=Path(os.environ.get("TRUE_SOFT_INPUT_FILE", PROJECT_ROOT / DEFAULT_INPUT_FILE)).resolve(),
        metadata_file=Path(os.environ.get("TRUE_SOFT_METADATA_FILE", PROJECT_ROOT / DEFAULT_METADATA_FILE)).resolve(),
        model_dir=model_dir,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        soft_label_file=Path(os.environ.get("TRUE_SOFT_LABEL_FILE", output_dir / "soft_labels.jsonl")).resolve(),
        hard_prediction_file=Path(
            os.environ.get("TRUE_SOFT_HARD_PREDICTION_FILE", output_dir / "hard_predictions.jsonl")
        ).resolve(),
        patterns_path=Path(os.environ.get("TRUE_SOFT_PATTERNS_PATH", PROJECT_ROOT / DEFAULT_PATTERNS_PATH)).resolve(),
        pattern_section=os.environ.get("TRUE_SOFT_PATTERN_SECTION", DEFAULT_PATTERN_SECTION),
        source_pattern_id=os.environ.get("TRUE_SOFT_SOURCE_PATTERN_ID"),
        inference_order=os.environ.get("TRUE_SOFT_INFERENCE_ORDER", DEFAULT_INFERENCE_ORDER),
        label_verbalizer=parse_label_verbalizer(os.environ.get("TRUE_SOFT_LABEL_VERBALIZER")),
        batch_size=int(os.environ.get("TRUE_SOFT_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        max_source_length=int(os.environ.get("TRUE_SOFT_MAX_SOURCE_LENGTH", DEFAULT_MAX_SOURCE_LENGTH)),
        max_generation_length=int(os.environ.get("TRUE_SOFT_MAX_GENERATION_LENGTH", DEFAULT_MAX_GENERATION_LENGTH)),
    )


def generate_true_soft_labels(config: TrueSoftConfig) -> dict[str, Any]:
    started_at = time.time()
    rows = read_jsonl(config.input_file)
    label_list = label_list_from_metadata(config.metadata_file)
    entity_types = entity_types_from_label_list(label_list)

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    candidate_token_ids = verbalizer_token_ids(tokenizer, config.label_verbalizer)
    source_pattern_id = config.source_pattern_id or source_pattern_from_training_summary(config.model_dir)
    if source_pattern_id is None:
        raise ValueError("Could not determine source pattern id from config or training_summary.json.")
    pattern = select_pattern(load_yaml(config.patterns_path), config.pattern_section, source_pattern_id)

    soft_rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []
    matched_entities = 0

    for batch_start in tqdm(range(0, len(rows), config.batch_size), desc="True soft labels", dynamic_ncols=True):
        batch = rows[batch_start : batch_start + config.batch_size]
        input_texts = [
            apply_member_wrapper(
                detokenize_tokens(row["tokens"]),
                pattern["template"],
                config.inference_order,
            )
            for row in batch
        ]
        tokenized = tokenizer(
            input_texts,
            max_length=config.max_source_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                max_length=config.max_generation_length,
                num_beams=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        sequences = generated.sequences.detach().cpu()
        score_tensors = [score.detach().cpu() for score in generated.scores]

        for batch_index, row in enumerate(batch):
            row_id = row.get("id", f"row-{batch_start + batch_index:06d}")
            generated_ids = generated_ids_for_scores(sequences[batch_index].tolist(), len(score_tensors))
            generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            soft_labels, entity_records = generated_true_soft_labels(
                tokenizer=tokenizer,
                tokens=row["tokens"],
                generated_text=generated_text,
                generated_ids=generated_ids,
                score_tensors=score_tensors,
                batch_index=batch_index,
                candidate_token_ids=candidate_token_ids,
                label_list=label_list,
                entity_types=entity_types,
            )
            predicted_tags = generated_text_to_bio_tags(generated_text, row["tokens"], entity_types)
            matched_entities += len(entity_records)
            common = {"id": row_id, "tokens": row["tokens"]}
            if "ner_tags" in row:
                common["ner_tags"] = row["ner_tags"]
                common["gold_tags"] = gold_tags_from_row(row, label_list)
            soft_rows.append({**common, "teacher": "verbalized_true_soft", "soft_labels": soft_labels})
            hard_rows.append(
                {
                    **common,
                    "predicted_tags": predicted_tags,
                    "generated_text": generated_text.strip(),
                    "entities": entity_records,
                }
            )

    write_jsonl(config.soft_label_file, soft_rows)
    write_jsonl(config.hard_prediction_file, hard_rows)
    manifest = {
        "teacher": "verbalized_true_soft",
        "input_file": repo_relative_path(config.input_file),
        "model_dir": repo_relative_path(config.model_dir),
        "checkpoint_dir": repo_relative_path(config.checkpoint_dir),
        "source_pattern_id": source_pattern_id,
        "source_template": pattern["template"],
        "inference_order": config.inference_order,
        "soft_label_file": repo_relative_path(config.soft_label_file),
        "hard_prediction_file": repo_relative_path(config.hard_prediction_file),
        "label_verbalizer": config.label_verbalizer,
        "candidate_token_ids": candidate_token_ids,
        "label_list": label_list,
        "rows_written": len(soft_rows),
        "matched_entities": matched_entities,
        "uncovered_token_policy": "hard_O_probability_1.0",
        "elapsed_seconds": time.time() - started_at,
    }
    write_json(config.output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    manifest = generate_true_soft_labels(load_config_from_env())
    print(
        "Finished verbalized true-soft generation: "
        f"rows={manifest['rows_written']} entities={manifest['matched_entities']} "
        f"soft={manifest['soft_label_file']}"
    )


if __name__ == "__main__":
    main()
