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
        repo_relative_path,
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
        repo_relative_path,
    )


FIRST_TO_LAST_ORDER = "first to last"
DEFAULT_INFERENCE_PATTERN_ID = "pattern_1_oada_original"
DEFAULT_INFERENCE_OUTPUT_DIR = "inference/first_to_last"
DEFAULT_SPLIT_FILES = {
    "mini_val": "data/interim/conll2003_kshot_bert/mini_val.jsonl",
    "validation": "data/interim/conll2003_kshot_bert/validation.jsonl",
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
        PERM=FIRST_TO_LAST_ORDER,
        sentence_text=sentence_text,
        order=FIRST_TO_LAST_ORDER,
    )


def select_inference_pattern(
    pattern_bank: dict,
    pattern_section: str,
    pattern_id: str,
) -> dict[str, str]:
    for pattern in _iter_pattern_bank(pattern_bank, pattern_section):
        if pattern["id"] == pattern_id:
            return pattern

    raise ValueError(f"Could not find inference pattern id: {pattern_id}")


def augment_inference_split(
    input_filepath: str,
    split_name: str,
    pattern: dict[str, str],
    id_to_string_map: dict,
    base_output_dir: str,
) -> dict[str, object]:
    """Write one shared first-to-last inference JSONL file for a split."""
    input_data = load_jsonl(input_filepath)
    output_dir = Path(base_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split_name}.jsonl"
    rows_written = 0

    with output_path.open("w", encoding="utf-8") as output_file:
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

    print(f"{split_name}: wrote {rows_written} rows to {output_path}")
    return {
        "source_file": repo_relative_path(input_filepath),
        "output_file": repo_relative_path(output_path),
        "source_rows": len(input_data),
        "generated_rows": rows_written,
    }


def main_orchestrator(
    split_files: dict[str, str],
    base_output_dir: str,
    pattern_bank: dict,
    id_to_string_map: dict,
    pattern_section: str = DEFAULT_PATTERN_SECTION,
) -> dict[str, dict[str, object]]:
    """Run inference augmentation for each configured evaluation split."""
    pattern_id = os.environ.get("AUG_INF_PATTERN_ID", DEFAULT_INFERENCE_PATTERN_ID)
    pattern = select_inference_pattern(pattern_bank, pattern_section, pattern_id)
    manifest: dict[str, object] = {
        "inference_order": FIRST_TO_LAST_ORDER,
        "source_pattern_id": pattern["id"],
        "source_pattern_type": pattern.get("type"),
        "source_template": pattern["template"],
        "splits": {},
    }

    for split_name, input_filepath in split_files.items():
        manifest["splits"][split_name] = augment_inference_split(
            input_filepath=input_filepath,
            split_name=split_name,
            pattern=pattern,
            id_to_string_map=id_to_string_map,
            base_output_dir=base_output_dir,
        )

    manifest_path = Path(base_output_dir) / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, indent=2, ensure_ascii=False)
    print(f"Wrote manifest to {manifest_path}")

    return manifest["splits"]


def _default_output_dir() -> Path:
    return PROJECT_ROOT / DEFAULT_OUTPUT_BASE_DIR / DEFAULT_INFERENCE_OUTPUT_DIR


def _split_files_from_env() -> dict[str, str]:
    return {
        split_name: os.environ.get(env_name, str(PROJECT_ROOT / default_filepath))
        for split_name, env_name, default_filepath in [
            ("mini_val", "AUG_INF_MINI_VAL_FILE", DEFAULT_SPLIT_FILES["mini_val"]),
            ("validation", "AUG_INF_VAL_FILE", DEFAULT_SPLIT_FILES["validation"]),
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

    total_rows = sum(split_record["generated_rows"] for split_record in rows_by_split.values())
    total_files = len(rows_by_split)
    print(f"Finished inference augmentation: {total_rows} rows across {total_files} files.")


if __name__ == "__main__":
    main()
