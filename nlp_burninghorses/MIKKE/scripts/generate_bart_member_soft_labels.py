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
import yaml

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
        generated_text_to_bio_tags,
        read_jsonl,
    )
    from .data_aug_train import _iter_pattern_bank, repo_relative_path
else:
    from bart import (
        BRACKETED_ENTITY_RE,
        detokenize_tokens,
        find_entity_span,
        generated_text_to_bio_tags,
        read_jsonl,
    )
    from data_aug_train import _iter_pattern_bank, repo_relative_path


DEFAULT_DATASET = "conll2003"
DEFAULT_METHOD = "pet_oada"
DEFAULT_MODEL_FAMILY = "bart_base"
DEFAULT_KSHOT_SPLIT = "k5_seed242"
DEFAULT_PATTERN = "pattern_01"
DEFAULT_INPUT_FILE = (
    "data/interim/conll2003_kshot_bert/k5_seed242/unlabeled_pool.jsonl"
)
DEFAULT_METADATA_FILE = "data/interim/conll2003_kshot_bert/metadata.json"
DEFAULT_MODEL_DIR = (
    "models/conll2003_kshot_seq2seq/bart_base/pet_oada/k5_seed242/pattern_01"
)
DEFAULT_CHECKPOINT_SUBDIR = "best_span_f1"
DEFAULT_PATTERNS_PATH = "nlp_burninghorses/MIKKE/scripts/patterns.yaml"
DEFAULT_PATTERN_SECTION = "patterns"
DEFAULT_INFERENCE_ORDER = "first to last"
DEFAULT_BATCH_SIZE = 16
DEFAULT_TEMPERATURE = 2.0
DEFAULT_MAX_SOURCE_LENGTH = 128
DEFAULT_MAX_GENERATION_LENGTH = 128
DEFAULT_NO_REPEAT_NGRAM_SIZE = 0
NON_CANDIDATE_LOGIT = -20.0


@dataclass(frozen=True)
class MemberSoftLabelConfig:
    dataset: str
    method: str
    model_family: str
    kshot_split: str
    pattern: str
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
    batch_size: int
    temperature: float
    hard_only: bool
    max_source_length: int
    max_generation_length: int
    no_repeat_ngram_size: int


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return yaml.safe_load(input_file)


def require_columns(rows: list[dict[str, Any]], columns: set[str], source_name: str) -> None:
    for row_index, row in enumerate(rows):
        missing = columns.difference(row)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{source_name} row {row_index} is missing: {missing_text}")


def label_list_from_metadata(metadata: dict[str, Any]) -> list[str]:
    label_list = metadata.get("label_list")
    if not isinstance(label_list, list) or not label_list:
        raise ValueError("Metadata must contain a non-empty label_list.")
    return [str(label) for label in label_list]


def entity_types_from_label_list(label_list: list[str]) -> list[str]:
    return sorted({label.split("-", 1)[1] for label in label_list if "-" in label})


def select_pattern(
    pattern_bank: dict[str, Any],
    pattern_section: str,
    source_pattern_id: str,
) -> dict[str, str]:
    for pattern in _iter_pattern_bank(pattern_bank, pattern_section):
        if pattern["id"] == source_pattern_id:
            return pattern

    raise ValueError(f"Could not find source pattern id: {source_pattern_id}")


def source_pattern_from_training_summary(model_dir: Path) -> str | None:
    summary_path = model_dir / "training_summary.json"
    if not summary_path.exists():
        return None

    summary = read_json(summary_path)
    experiment = summary.get("experiment", {})
    source_pattern_id = experiment.get("source_pattern_id") or summary.get("source_pattern_id")
    return str(source_pattern_id) if source_pattern_id else None


def apply_member_wrapper(sentence_text: str, pattern_template: str, inference_order: str) -> str:
    return pattern_template.format(
        SEN=sentence_text,
        PERM=inference_order,
        sentence_text=sentence_text,
        order=inference_order,
    )


def class_token_ids(tokenizer, entity_type: str) -> set[int]:
    ids: set[int] = set()
    for text in (entity_type, f" {entity_type}"):
        ids.update(tokenizer.encode(text, add_special_tokens=False))
    return ids


def build_entity_type_token_map(tokenizer, entity_types: list[str]) -> dict[str, set[int]]:
    return {entity_type: class_token_ids(tokenizer, entity_type) for entity_type in entity_types}


def generated_ids_for_scores(sequence: list[int], score_steps: int) -> list[int]:
    if len(sequence) == score_steps:
        return sequence
    if len(sequence) == score_steps + 1:
        return sequence[1:]
    return sequence[-score_steps:]


def generation_step_offsets(tokenizer, generated_ids: list[int]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    previous_text = ""

    for index in range(len(generated_ids)):
        current_text = tokenizer.decode(
            generated_ids[: index + 1],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        offsets.append((len(previous_text), len(current_text)))
        previous_text = current_text

    return offsets


def find_generation_step_for_span(
    step_offsets: list[tuple[int, int]],
    char_span: tuple[int, int],
) -> int | None:
    span_start, span_end = char_span
    for step_index, (step_start, step_end) in enumerate(step_offsets):
        if step_end <= span_start or step_start >= span_end:
            continue
        return step_index
    return None


def project_entity_type_scores(
    step_scores: torch.Tensor | None,
    entity_type_token_ids: dict[str, set[int]],
    predicted_entity_type: str,
) -> dict[str, float]:
    if step_scores is None:
        return {
            entity_type: 0.0 if entity_type == predicted_entity_type else NON_CANDIDATE_LOGIT
            for entity_type in entity_type_token_ids
        }

    projected_scores: dict[str, float] = {}
    for entity_type, token_ids in entity_type_token_ids.items():
        if not token_ids:
            projected_scores[entity_type] = NON_CANDIDATE_LOGIT
            continue
        projected_scores[entity_type] = max(float(step_scores[token_id]) for token_id in token_ids)
    return projected_scores


def softmax(values: list[float], temperature: float) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float32) / temperature
    probabilities = torch.softmax(tensor, dim=-1)
    return [float(value) for value in probabilities.tolist()]


def scores_to_tags(label_scores: list[list[float]], label_list: list[str]) -> list[str]:
    tags: list[str] = []
    for token_scores in label_scores:
        best_index = max(range(len(token_scores)), key=token_scores.__getitem__)
        tags.append(label_list[best_index])
    return tags


def gold_tags_from_row(row: dict[str, Any], label_list: list[str]) -> list[str] | None:
    ner_tags = row.get("ner_tags")
    if ner_tags is None:
        return None

    gold_tags: list[str] = []
    for tag in ner_tags:
        if isinstance(tag, int):
            gold_tags.append(label_list[tag])
        elif isinstance(tag, str) and tag.isdigit():
            gold_tags.append(label_list[int(tag)])
        else:
            gold_tags.append(str(tag))
    return gold_tags


def generation_to_label_scores(
    tokens: list[str],
    generated_text: str,
    step_offsets: list[tuple[int, int]],
    step_scores: list[torch.Tensor],
    label_list: list[str],
    entity_types: list[str],
    entity_type_token_ids: dict[str, set[int]],
) -> list[list[float]]:
    label_to_index = {label: index for index, label in enumerate(label_list)}
    label_scores = [[NON_CANDIDATE_LOGIT] * len(label_list) for _ in tokens]
    occupied_indices: set[int] = set()

    if "O" not in label_to_index:
        raise ValueError("label_list must include the O class.")

    for token_scores in label_scores:
        token_scores[label_to_index["O"]] = 0.0

    for match in BRACKETED_ENTITY_RE.finditer(generated_text):
        entity_text = match.group(1).strip()
        entity_type = match.group(2).strip()
        if entity_type not in entity_types:
            continue

        span = find_entity_span(tokens, entity_text, occupied_indices)
        if span is None:
            continue

        generation_step = find_generation_step_for_span(step_offsets, match.span(2))
        coarse_scores = project_entity_type_scores(
            step_scores[generation_step] if generation_step is not None else None,
            entity_type_token_ids,
            entity_type,
        )

        start_index, end_index = span
        for token_index in range(start_index, end_index):
            prefix = "B" if token_index == start_index else "I"
            token_scores = [NON_CANDIDATE_LOGIT] * len(label_list)
            for candidate_type, score in coarse_scores.items():
                label = f"{prefix}-{candidate_type}"
                if label in label_to_index:
                    token_scores[label_to_index[label]] = score
            label_scores[token_index] = token_scores

        occupied_indices.update(range(start_index, end_index))

    return label_scores


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(
    path: Path,
    config: MemberSoftLabelConfig,
    pattern: dict[str, str],
    label_list: list[str],
    rows_written: int,
    elapsed_seconds: float,
) -> None:
    manifest = {
        "dataset": config.dataset,
        "method": config.method,
        "model_family": config.model_family,
        "kshot_split": config.kshot_split,
        "pattern": config.pattern,
        "source_pattern_id": pattern["id"],
        "source_pattern_type": pattern.get("type"),
        "source_template": pattern["template"],
        "inference_order": config.inference_order,
        "input_file": repo_relative_path(config.input_file),
        "metadata_file": repo_relative_path(config.metadata_file),
        "model_dir": repo_relative_path(config.model_dir),
        "checkpoint_dir": repo_relative_path(config.checkpoint_dir),
        "output_dir": repo_relative_path(config.output_dir),
        "soft_label_file": repo_relative_path(config.soft_label_file),
        "hard_prediction_file": repo_relative_path(config.hard_prediction_file),
        "label_list": label_list,
        "temperature": config.temperature,
        "hard_only": config.hard_only,
        "rows_written": rows_written,
        "elapsed_seconds": elapsed_seconds,
        "score_projection": (
            "BART generation-step vocabulary logits projected onto BIO labels. "
            "Tokens not aligned to a generated entity receive the O-class score."
        ),
    }
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, indent=2, ensure_ascii=False)


def generate_member_soft_labels(config: MemberSoftLabelConfig) -> dict[str, Any]:
    started_at = time.time()
    rows = read_jsonl(config.input_file)
    metadata = read_json(config.metadata_file)
    require_columns(rows, {"tokens"}, str(config.input_file))

    label_list = label_list_from_metadata(metadata)
    entity_types = entity_types_from_label_list(label_list)
    source_pattern_id = (
        config.source_pattern_id
        or source_pattern_from_training_summary(config.model_dir)
    )
    if source_pattern_id is None:
        raise ValueError(
            "Could not determine source pattern id. Set SOFT_LABEL_SOURCE_PATTERN_ID "
            "or keep training_summary.json in the model directory."
        )

    pattern = select_pattern(
        load_yaml(config.patterns_path),
        config.pattern_section,
        source_pattern_id,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.checkpoint_dir)
    tokenizer.model_max_length = config.max_source_length
    model.generation_config.no_repeat_ngram_size = config.no_repeat_ngram_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    entity_type_token_ids = build_entity_type_token_map(tokenizer, entity_types)
    soft_label_rows: list[dict[str, Any]] = []
    hard_prediction_rows: list[dict[str, Any]] = []

    for batch_start in tqdm(
        range(0, len(rows), config.batch_size),
        desc="BART member soft labels",
        dynamic_ncols=True,
    ):
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
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                return_dict_in_generate=True,
                output_scores=not config.hard_only,
            )

        sequences = generated.sequences.detach().cpu()
        score_tensors = [] if config.hard_only else [score.detach().cpu() for score in generated.scores]

        for batch_index, row in enumerate(batch):
            sequence_ids = sequences[batch_index].tolist()
            score_step_count = len(score_tensors) if score_tensors else max(0, len(sequence_ids) - 1)
            generated_ids = generated_ids_for_scores(
                sequence_ids,
                score_step_count,
            )
            generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            row_id = row.get("id", f"row-{batch_start + batch_index:06d}")
            generated_tags = generated_text_to_bio_tags(
                generated_text,
                row["tokens"],
                set(entity_types),
            )

            hard_prediction_row: dict[str, Any] = {
                "id": row_id,
                "tokens": row["tokens"],
                "predicted_tags": generated_tags,
                "generated_text": generated_text.strip(),
            }
            if not config.hard_only:
                step_offsets = generation_step_offsets(tokenizer, generated_ids)
                token_step_scores = [
                    score_tensor[batch_index] for score_tensor in score_tensors[: len(generated_ids)]
                ]
                label_scores = generation_to_label_scores(
                    tokens=row["tokens"],
                    generated_text=generated_text,
                    step_offsets=step_offsets,
                    step_scores=token_step_scores,
                    label_list=label_list,
                    entity_types=entity_types,
                    entity_type_token_ids=entity_type_token_ids,
                )
                soft_labels = [
                    softmax(token_scores, config.temperature) for token_scores in label_scores
                ]
                soft_label_rows.append(
                    {"id": row_id, "tokens": row["tokens"], "soft_labels": soft_labels}
                )
                hard_prediction_row["score_argmax_tags"] = scores_to_tags(label_scores, label_list)
            if "ner_tags" in row:
                hard_prediction_row["ner_tags"] = row["ner_tags"]
                hard_prediction_row["gold_tags"] = gold_tags_from_row(row, label_list)
            hard_prediction_rows.append(hard_prediction_row)

    if not config.hard_only:
        write_jsonl(config.soft_label_file, soft_label_rows)
    write_jsonl(config.hard_prediction_file, hard_prediction_rows)
    elapsed_seconds = time.time() - started_at
    write_manifest(
        config.output_dir / "manifest.json",
        config,
        pattern,
        label_list,
        len(hard_prediction_rows),
        elapsed_seconds,
    )

    return {
        "rows_written": len(hard_prediction_rows),
        "soft_label_file": repo_relative_path(config.soft_label_file),
        "hard_prediction_file": repo_relative_path(config.hard_prediction_file),
        "elapsed_seconds": elapsed_seconds,
    }


def _path_from_env(name: str, default_relative_path: str) -> Path:
    value = os.environ.get(name)
    if value is not None:
        return Path(value).expanduser().resolve()
    return PROJECT_ROOT / default_relative_path


def bool_from_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config_from_env() -> MemberSoftLabelConfig:
    model_dir = _path_from_env("SOFT_LABEL_MODEL_DIR", DEFAULT_MODEL_DIR)
    checkpoint_dir = Path(
        os.environ.get("SOFT_LABEL_CHECKPOINT_DIR", model_dir / DEFAULT_CHECKPOINT_SUBDIR)
    ).expanduser().resolve()
    legacy_output_file = os.environ.get("SOFT_LABEL_OUTPUT_FILE")
    if legacy_output_file is not None:
        soft_label_file = Path(legacy_output_file).expanduser().resolve()
        output_dir = soft_label_file.parent
    else:
        output_dir = Path(
            os.environ.get("SOFT_LABEL_OUTPUT_DIR", model_dir / "soft_labels" / "unlabeled_pool")
        ).expanduser().resolve()
        soft_label_file = output_dir / "soft_labels.jsonl"

    hard_prediction_file = Path(
        os.environ.get("SOFT_LABEL_HARD_PREDICTION_FILE", output_dir / "hard_predictions.jsonl")
    ).expanduser().resolve()

    return MemberSoftLabelConfig(
        dataset=os.environ.get("SOFT_LABEL_DATASET", DEFAULT_DATASET),
        method=os.environ.get("SOFT_LABEL_METHOD", DEFAULT_METHOD),
        model_family=os.environ.get("SOFT_LABEL_MODEL_FAMILY", DEFAULT_MODEL_FAMILY),
        kshot_split=os.environ.get("SOFT_LABEL_KSHOT_SPLIT", DEFAULT_KSHOT_SPLIT),
        pattern=os.environ.get("SOFT_LABEL_PATTERN", DEFAULT_PATTERN),
        input_file=_path_from_env("SOFT_LABEL_INPUT_FILE", DEFAULT_INPUT_FILE),
        metadata_file=_path_from_env("SOFT_LABEL_METADATA_FILE", DEFAULT_METADATA_FILE),
        model_dir=model_dir,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        soft_label_file=soft_label_file,
        hard_prediction_file=hard_prediction_file,
        patterns_path=_path_from_env("SOFT_LABEL_PATTERNS_PATH", DEFAULT_PATTERNS_PATH),
        pattern_section=os.environ.get("SOFT_LABEL_PATTERN_SECTION", DEFAULT_PATTERN_SECTION),
        source_pattern_id=os.environ.get("SOFT_LABEL_SOURCE_PATTERN_ID"),
        inference_order=os.environ.get("SOFT_LABEL_INFERENCE_ORDER", DEFAULT_INFERENCE_ORDER),
        batch_size=int(os.environ.get("SOFT_LABEL_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        temperature=float(os.environ.get("SOFT_LABEL_TEMPERATURE", DEFAULT_TEMPERATURE)),
        hard_only=bool_from_env("SOFT_LABEL_HARD_ONLY", default=False),
        max_source_length=int(os.environ.get("SOFT_LABEL_MAX_SOURCE_LENGTH", DEFAULT_MAX_SOURCE_LENGTH)),
        max_generation_length=int(
            os.environ.get("SOFT_LABEL_MAX_GENERATION_LENGTH", DEFAULT_MAX_GENERATION_LENGTH)
        ),
        no_repeat_ngram_size=int(
            os.environ.get("SOFT_LABEL_NO_REPEAT_NGRAM_SIZE", DEFAULT_NO_REPEAT_NGRAM_SIZE)
        ),
    )


def main() -> None:
    config = load_config_from_env()
    summary = generate_member_soft_labels(config)
    print(
        "Finished BART member soft-label generation: "
        f"rows={summary['rows_written']} "
        f"soft={summary['soft_label_file']} "
        f"hard={summary['hard_prediction_file']} "
        f"time_min={summary['elapsed_seconds'] / 60:.2f}"
    )


if __name__ == "__main__":
    main()
