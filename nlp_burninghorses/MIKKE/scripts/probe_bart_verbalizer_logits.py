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

from nlp_burninghorses.MIKKE.scripts.bart import BRACKETED_ENTITY_RE, normalize_entity_type, read_jsonl  # noqa: E402
from nlp_burninghorses.MIKKE.scripts.generate_bart_member_soft_labels import (  # noqa: E402
    generation_step_offsets,
    generated_ids_for_scores,
)

DEFAULT_MODEL_DIR = (
    "models/conll2003_kshot_seq2seq/bart_base/pet_oada_verbalized/"
    "k5_seed42/pattern_01_xe_verbalized_pilot1000"
)
DEFAULT_CHECKPOINT_SUBDIR = "final_step"
DEFAULT_INPUT_FILE = "data/interim/conll2003_kshot_seq2seq/inference/first_to_last_verbalized/validation.jsonl"
DEFAULT_GOLD_FILE = "data/interim/conll2003_kshot_bert/validation.jsonl"
DEFAULT_OUTPUT_FILE = (
    "reports/verbalizer_logits/"
    "k5_seed42_pattern_01_xe_verbalized_pilot1000_validation.jsonl"
)
DEFAULT_LABEL_VERBALIZER = {
    "PER": "person",
    "LOC": "location",
    "ORG": "organization",
    "MISC": "other",
}
DEFAULT_MAX_ROWS = 200
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_SOURCE_LENGTH = 128
DEFAULT_MAX_GENERATION_LENGTH = 128


@dataclass(frozen=True)
class VerbalizerProbeConfig:
    checkpoint_dir: Path
    input_file: Path
    gold_file: Path
    output_file: Path
    label_verbalizer: dict[str, str]
    max_rows: int
    batch_size: int
    max_source_length: int
    max_generation_length: int


def parse_label_verbalizer(raw_value: str | None) -> dict[str, str]:
    if raw_value is None:
        return DEFAULT_LABEL_VERBALIZER.copy()

    verbalizer: dict[str, str] = {}
    for item in raw_value.split(","):
        if not item.strip():
            continue
        source, separator, target = item.partition("=")
        if not separator:
            raise ValueError("Verbalizer entries must use TYPE=word syntax.")
        verbalizer[source.strip()] = target.strip()
    return verbalizer


def repo_relative_path(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved_path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)


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


def verbalizer_token_ids(tokenizer, label_verbalizer: dict[str, str]) -> dict[str, list[list[int]]]:
    token_ids: dict[str, list[list[int]]] = {}
    for label, verbalizer in label_verbalizer.items():
        candidate_ids: list[list[int]] = []
        for text in (verbalizer, f" {verbalizer}"):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids and ids not in candidate_ids:
                candidate_ids.append(ids)
        if not candidate_ids:
            raise ValueError(f"Verbalizer {label}={verbalizer!r} has no tokenized form.")
        token_ids[label] = candidate_ids
    return token_ids


def candidate_probabilities(
    step_scores: torch.Tensor,
    candidate_token_ids: dict[str, list[list[int]]],
) -> dict[str, float]:
    selected_logits = torch.tensor(
        [
            max(float(step_scores[token_sequence[0]]) for token_sequence in token_sequences)
            for token_sequences in candidate_token_ids.values()
        ],
        dtype=torch.float32,
    )
    probabilities = torch.softmax(selected_logits, dim=0).tolist()
    return {
        label: float(probability)
        for label, probability in zip(candidate_token_ids, probabilities, strict=True)
    }


def label_for_token_id(
    token_id: int | None,
    candidate_token_ids: dict[str, list[list[int]]],
) -> str | None:
    if token_id is None:
        return None
    for label, token_sequences in candidate_token_ids.items():
        if any(token_id == token_sequence[0] for token_sequence in token_sequences):
            return label
    return None


def candidate_logit_for_label(
    step_scores: torch.Tensor,
    label: str,
    candidate_token_ids: dict[str, list[list[int]]],
) -> float:
    return max(float(step_scores[token_sequence[0]]) for token_sequence in candidate_token_ids[label])


def aligned_score_step(
    score_tensors: list[torch.Tensor],
    batch_index: int,
    generated_token_id: int | None,
    decoded_step_index: int,
    candidate_token_ids: dict[str, list[list[int]]],
) -> int:
    generated_label = label_for_token_id(generated_token_id, candidate_token_ids)
    if generated_label is None:
        return decoded_step_index

    candidate_steps = [
        step_index
        for step_index in range(decoded_step_index - 2, decoded_step_index + 3)
        if 0 <= step_index < len(score_tensors)
    ]
    return max(
        candidate_steps,
        key=lambda step_index: candidate_logit_for_label(
            score_tensors[step_index][batch_index],
            generated_label,
            candidate_token_ids,
        ),
    )


def load_config_from_env() -> VerbalizerProbeConfig:
    model_dir = Path(os.environ.get("VERBALIZER_PROBE_MODEL_DIR", PROJECT_ROOT / DEFAULT_MODEL_DIR))
    checkpoint_dir = Path(
        os.environ.get("VERBALIZER_PROBE_CHECKPOINT_DIR", model_dir / DEFAULT_CHECKPOINT_SUBDIR)
    ).expanduser().resolve()
    return VerbalizerProbeConfig(
        checkpoint_dir=checkpoint_dir,
        input_file=Path(os.environ.get("VERBALIZER_PROBE_INPUT_FILE", PROJECT_ROOT / DEFAULT_INPUT_FILE)).resolve(),
        gold_file=Path(os.environ.get("VERBALIZER_PROBE_GOLD_FILE", PROJECT_ROOT / DEFAULT_GOLD_FILE)).resolve(),
        output_file=Path(os.environ.get("VERBALIZER_PROBE_OUTPUT_FILE", PROJECT_ROOT / DEFAULT_OUTPUT_FILE)).resolve(),
        label_verbalizer=parse_label_verbalizer(os.environ.get("VERBALIZER_PROBE_LABEL_VERBALIZER")),
        max_rows=int(os.environ.get("VERBALIZER_PROBE_MAX_ROWS", DEFAULT_MAX_ROWS)),
        batch_size=int(os.environ.get("VERBALIZER_PROBE_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        max_source_length=int(os.environ.get("VERBALIZER_PROBE_MAX_SOURCE_LENGTH", DEFAULT_MAX_SOURCE_LENGTH)),
        max_generation_length=int(os.environ.get("VERBALIZER_PROBE_MAX_GENERATION_LENGTH", DEFAULT_MAX_GENERATION_LENGTH)),
    )


def run_probe(config: VerbalizerProbeConfig) -> dict[str, Any]:
    started_at = time.time()
    inference_rows = read_jsonl(config.input_file)[: config.max_rows]
    gold_rows = read_jsonl(config.gold_file)[: config.max_rows]
    if len(inference_rows) != len(gold_rows):
        raise ValueError("Input and gold rows must have the same row count after max_rows slicing.")

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.checkpoint_dir)
    model.generation_config.no_repeat_ngram_size = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    candidate_token_ids = verbalizer_token_ids(tokenizer, config.label_verbalizer)
    probe_rows: list[dict[str, Any]] = []
    matched_entities = 0

    for batch_start in tqdm(
        range(0, len(inference_rows), config.batch_size),
        desc="Probe verbalizer logits",
        dynamic_ncols=True,
    ):
        batch = inference_rows[batch_start : batch_start + config.batch_size]
        tokenized = tokenizer(
            [row["input_text"] for row in batch],
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
            row_index = batch_start + batch_index
            generated_ids = generated_ids_for_scores(sequences[batch_index].tolist(), len(score_tensors))
            generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            step_offsets = generation_step_offsets(tokenizer, generated_ids)
            entities: list[dict[str, Any]] = []

            for match in BRACKETED_ENTITY_RE.finditer(generated_text):
                surface_label = match.group(2).strip()
                canonical_label = normalize_entity_type(surface_label)
                step_index = find_generation_step_for_span(step_offsets, match.span(2))
                if step_index is None:
                    continue
                matched_entities += 1
                generated_token_id = generated_ids[step_index] if step_index < len(generated_ids) else None
                score_step = aligned_score_step(
                    score_tensors,
                    batch_index,
                    generated_token_id,
                    step_index,
                    candidate_token_ids,
                )
                entities.append(
                    {
                        "entity_text": match.group(1).strip(),
                        "surface_label": surface_label,
                        "canonical_label": canonical_label,
                        "decoded_type_step": step_index,
                        "score_step": score_step,
                        "generated_token_id": generated_token_id,
                        "generated_token": (
                            tokenizer.convert_ids_to_tokens([generated_token_id])[0]
                            if generated_token_id is not None
                            else None
                        ),
                        "candidate_token_ids": candidate_token_ids,
                        "label_probabilities": candidate_probabilities(
                            score_tensors[score_step][batch_index],
                            candidate_token_ids,
                        ),
                    }
                )

            probe_rows.append(
                {
                    "id": f"row-{row_index:06d}",
                    "tokens": gold_rows[row_index]["tokens"],
                    "generated_text": generated_text.strip(),
                    "entities": entities,
                }
            )

    write_jsonl(config.output_file, probe_rows)
    manifest = {
        "model_checkpoint": repo_relative_path(config.checkpoint_dir),
        "input_file": repo_relative_path(config.input_file),
        "gold_file": repo_relative_path(config.gold_file),
        "output_file": repo_relative_path(config.output_file),
        "label_verbalizer": config.label_verbalizer,
        "candidate_token_ids": candidate_token_ids,
        "rows_written": len(probe_rows),
        "matched_entities": matched_entities,
        "elapsed_seconds": time.time() - started_at,
    }
    write_json(config.output_file.with_suffix(".manifest.json"), manifest)
    return manifest


def main() -> None:
    config = load_config_from_env()
    manifest = run_probe(config)
    print(
        "Finished verbalizer logit probe: "
        f"rows={manifest['rows_written']} entities={manifest['matched_entities']} "
        f"output={manifest['output_file']}"
    )


if __name__ == "__main__":
    main()
