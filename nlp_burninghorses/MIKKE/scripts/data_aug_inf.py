import json
import os
from pathlib import Path

if __package__:
    from .data_aug_train import (
        DEFAULT_DATASET,
        DEFAULT_OUTPUT_BASE_DIR,
        DEFAULT_PATTERN_SECTION,
        PROJECT_ROOT,
        SCRIPT_DIR,
        _iter_pattern_bank,
        _label_to_string,
        detokenize_tokens,
        load_jsonl,
        load_yaml,
    )
else:
    from data_aug_train import (
        DEFAULT_DATASET,
        DEFAULT_OUTPUT_BASE_DIR,
        DEFAULT_PATTERN_SECTION,
        PROJECT_ROOT,
        SCRIPT_DIR,
        _iter_pattern_bank,
        _label_to_string,
        detokenize_tokens,
        load_jsonl,
        load_yaml,
    )


LEFT_TO_RIGHT_ORDER = "left to right"
DEFAULT_SPLIT_FILES = {
    "mini-val": "data/interim/conll2003_kshot_bert/mini_val.jsonl",
    "val": "data/interim/conll2003_kshot_bert/validation.jsonl",
    "test": "data/interim/conll2003_kshot_bert/test.jsonl",
}


def extract_entities_left_to_right(
    tokens: list[str],
    ner_tags: list[int],
    id_to_string_map: dict,
) -> list[tuple[str, str]]:
    """Extract contiguous BIO entities in sentence order."""
    if len(tokens) != len(ner_tags):
        raise ValueError("tokens and ner_tags must have the same length.")

    sequential_entities: list[tuple[str, str]] = []
    current_type: str | None = None
    current_tokens: list[str] = []

    def flush_current_entity() -> None:
        nonlocal current_type, current_tokens
        if current_type is not None and current_tokens:
            sequential_entities.append((detokenize_tokens(current_tokens), current_type))
        current_type = None
        current_tokens = []

    for token, tag_id in zip(tokens, ner_tags):
        tag = _label_to_string(tag_id, id_to_string_map)

        if tag == "O" or "-" not in tag:
            flush_current_entity()
            continue

        prefix, entity_type = tag.split("-", 1)

        if prefix == "B" or current_type != entity_type:
            flush_current_entity()
            current_type = entity_type
            current_tokens = [token]
        elif prefix == "I":
            current_tokens.append(token)
        else:
            flush_current_entity()

    flush_current_entity()
    return sequential_entities


def generate_inference_target(sequential_entities: list[tuple[str, str]]) -> str:
    """Build the dense bracket target in left-to-right entity order."""
    return " ".join(
        f"[{entity_text}]{entity_type}" for entity_text, entity_type in sequential_entities
    )


def apply_inference_pet_wrapper(sentence_text: str, pattern_template: str) -> str:
    """Inject the sentence and the fixed inference order into a PET pattern."""
    return pattern_template.format(
        SEN=sentence_text,
        PERM=LEFT_TO_RIGHT_ORDER,
        sentence_text=sentence_text,
        order=LEFT_TO_RIGHT_ORDER,
    )


def augment_inference_split(
    input_filepath: str,
    split_name: str,
    pattern_bank: list[dict[str, str]],
    id_to_string_map: dict,
    base_output_dir: str,
) -> dict[str, int]:
    """Write one inference JSONL file per PET pattern for a split."""
    input_data = load_jsonl(input_filepath)
    split_output_dir = Path(base_output_dir) / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_pattern: dict[str, int] = {}

    for pattern in pattern_bank:
        pattern_id = pattern["id"]
        pattern_output_path = split_output_dir / f"{pattern_id}.jsonl"
        rows_written = 0

        with pattern_output_path.open("w", encoding="utf-8") as output_file:
            for example in input_data:
                tokens = example["tokens"]
                ner_tags = example["ner_tags"]
                sentence_text = detokenize_tokens(tokens)
                sequential_entities = extract_entities_left_to_right(
                    tokens,
                    ner_tags,
                    id_to_string_map,
                )
                row = {
                    "input_text": apply_inference_pet_wrapper(
                        sentence_text,
                        pattern["template"],
                    ),
                    "target_text": generate_inference_target(sequential_entities),
                }
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1

        rows_by_pattern[pattern_id] = rows_written
        print(
            f"{split_name}/{pattern_id}: wrote {rows_written} rows to {pattern_output_path}"
        )

    return rows_by_pattern


def main_orchestrator(
    split_files: dict[str, str],
    base_output_dir: str,
    pattern_bank: dict,
    id_to_string_map: dict,
    pattern_section: str = DEFAULT_PATTERN_SECTION,
) -> dict[str, dict[str, int]]:
    """Run inference augmentation for each configured evaluation split."""
    patterns = _iter_pattern_bank(pattern_bank, pattern_section)
    rows_by_split: dict[str, dict[str, int]] = {}

    for split_name, input_filepath in split_files.items():
        rows_by_split[split_name] = augment_inference_split(
            input_filepath=input_filepath,
            split_name=split_name,
            pattern_bank=patterns,
            id_to_string_map=id_to_string_map,
            base_output_dir=base_output_dir,
        )

    return rows_by_split


def _default_output_dir() -> Path:
    return PROJECT_ROOT / DEFAULT_OUTPUT_BASE_DIR


def _split_files_from_env() -> dict[str, str]:
    return {
        split_name: os.environ.get(env_name, str(PROJECT_ROOT / default_filepath))
        for split_name, env_name, default_filepath in [
            ("mini-val", "AUG_INF_MINI_VAL_FILE", DEFAULT_SPLIT_FILES["mini-val"]),
            ("val", "AUG_INF_VAL_FILE", DEFAULT_SPLIT_FILES["val"]),
            ("test", "AUG_INF_TEST_FILE", DEFAULT_SPLIT_FILES["test"]),
        ]
    }


def main() -> None:
    patterns_path = Path(os.environ.get("AUG_PATTERNS_PATH", SCRIPT_DIR / "patterns.yaml"))
    mapping_path = Path(os.environ.get("AUG_MAPPING_PATH", SCRIPT_DIR.parent / "mapping.yaml"))
    dataset_name = os.environ.get("AUG_DATASET", DEFAULT_DATASET)
    pattern_section = os.environ.get("AUG_PATTERN_SECTION", DEFAULT_PATTERN_SECTION)
    output_dir = Path(os.environ.get("AUG_INF_OUTPUT_DIR", _default_output_dir()))

    pattern_bank = load_yaml(patterns_path)
    id_to_string_map = load_yaml(mapping_path)[dataset_name]["id_to_label"]
    rows_by_split = main_orchestrator(
        split_files=_split_files_from_env(),
        base_output_dir=str(output_dir),
        pattern_bank=pattern_bank,
        id_to_string_map=id_to_string_map,
        pattern_section=pattern_section,
    )

    total_rows = sum(sum(rows_by_pattern.values()) for rows_by_pattern in rows_by_split.values())
    total_files = sum(len(rows_by_pattern) for rows_by_pattern in rows_by_split.values())
    print(f"Finished inference augmentation: {total_rows} rows across {total_files} files.")


if __name__ == "__main__":
    main()
