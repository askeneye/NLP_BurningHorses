import itertools
import json
import os
import random
from pathlib import Path

import yaml


PERMUTATION_SAMPLE_SIZE = 20
DEFAULT_DATASET = "conll2003"
DEFAULT_FEWSHOT_DIR = "data/interim/conll2003_kshot_bert/k5_seed242"
DEFAULT_OUTPUT_BASE_DIR = "data/interim/conll2003_kshot_pet_oada"
DEFAULT_PATTERN_SECTION = "patterns"

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start_path: Path) -> Path:
    for path in [start_path, *start_path.parents]:
        if (path / "pyproject.toml").exists():
            return path

    raise FileNotFoundError("Could not locate project root containing pyproject.toml.")


PROJECT_ROOT = find_project_root(SCRIPT_DIR)


def get_oada_permutations(entity_schema: list[str]) -> list[tuple[str, ...]]:
    """Return OADA permutations from the global entity schema."""
    if len(entity_schema) <= 4:
        return list(itertools.permutations(entity_schema))

    all_permutations = list(itertools.permutations(entity_schema))
    return random.sample(all_permutations, PERMUTATION_SAMPLE_SIZE)


def _label_to_string(label_id: int | str, id_to_string_map: dict) -> str:
    if isinstance(label_id, str) and not label_id.isdigit():
        return label_id

    return id_to_string_map.get(label_id, id_to_string_map.get(int(label_id), str(label_id)))


def detokenize_tokens(tokens: list[str]) -> str:
    """Reconstruct readable text from CoNLL-style tokenized words."""
    no_space_before = {
        ".",
        ",",
        ":",
        ";",
        "?",
        "!",
        "%",
        ")",
        "]",
        "}",
        "'s",
        "'m",
        "'re",
        "'ve",
        "'ll",
        "'d",
        "n't",
    }
    no_space_after = {"(", "[", "{", "$", "#"}
    quote_is_open = False
    previous_opened_quote = False
    text = ""

    for token in tokens:
        if token in {'"', "``", "''"}:
            if token == "''" or (token == '"' and quote_is_open):
                text += '"'
                quote_is_open = False
                previous_opened_quote = False
            else:
                if text and not text.endswith(" "):
                    text += " "
                text += '"'
                quote_is_open = True
                previous_opened_quote = True
            continue

        if not text:
            text = token
        elif token in no_space_before or token.startswith("'"):
            text += token
        elif text[-1] in no_space_after or previous_opened_quote:
            text += token
        else:
            text += f" {token}"

        previous_opened_quote = False

    return text


def extract_and_group_entities(
    tokens: list[str],
    ner_tags: list[int],
    id_to_string_map: dict,
) -> dict[str, list[str]]:
    """Extract contiguous BIO entities and group them by coarse entity type."""
    if len(tokens) != len(ner_tags):
        raise ValueError("tokens and ner_tags must have the same length.")

    entities_by_type: dict[str, list[str]] = {}
    current_type: str | None = None
    current_tokens: list[str] = []

    def flush_current_entity() -> None:
        nonlocal current_type, current_tokens
        if current_type is not None and current_tokens:
            entities_by_type.setdefault(current_type, []).append(detokenize_tokens(current_tokens))
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
    return entities_by_type


def generate_oada_target(
    entities_by_type: dict[str, list[str]],
    order: tuple[str, ...],
) -> str:
    """Build target text in the requested entity-type order."""
    target_parts: list[str] = []

    for entity_type in order:
        for entity_text in entities_by_type.get(entity_type, []):
            target_parts.append(f"[{entity_text}]{entity_type}")

    return " ".join(target_parts)


def apply_pet_wrapper(
    sentence_text: str,
    order: tuple[str, ...],
    pattern_template: str,
) -> str:
    """Inject a sentence and OADA order into a PET pattern template."""
    order_text = ", ".join(order)
    return pattern_template.format(
        SEN=sentence_text,
        PERM=order_text,
        sentence_text=sentence_text,
        order=order_text,
    )


def augment_data(
    input_data: list[dict],
    pattern_id: str,
    pattern_template: str,
    schema: list[str],
    id_to_string_map: dict,
    output_filepath: str,
) -> int:
    """Write augmented BART JSONL rows for a single PET pattern."""
    output_path = Path(output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    permutations = get_oada_permutations(schema)
    rows_written = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for example in input_data:
            tokens = example["tokens"]
            ner_tags = example["ner_tags"]
            sentence_text = detokenize_tokens(tokens)
            entities_by_type = extract_and_group_entities(tokens, ner_tags, id_to_string_map)

            for order in permutations:
                row = {
                    "input_text": apply_pet_wrapper(sentence_text, order, pattern_template),
                    "target_text": generate_oada_target(entities_by_type, order),
                }
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1

    print(f"{pattern_id}: wrote {rows_written} rows to {output_path}")
    return rows_written


def load_jsonl(filepath: str | Path) -> list[dict]:
    rows: list[dict] = []

    with Path(filepath).open("r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped_line = line.strip()
            if stripped_line:
                rows.append(json.loads(stripped_line))

    return rows


def load_yaml(filepath: str | Path) -> dict:
    with Path(filepath).open("r", encoding="utf-8") as input_file:
        return yaml.safe_load(input_file)


def _iter_pattern_bank(
    pattern_bank: dict | list[dict[str, str]],
    pattern_section: str = DEFAULT_PATTERN_SECTION,
) -> list[dict[str, str]]:
    if isinstance(pattern_bank, list):
        patterns = pattern_bank
    else:
        patterns = pattern_bank.get(
            pattern_section,
            pattern_bank.get(DEFAULT_PATTERN_SECTION, pattern_bank),
        )

    if isinstance(patterns, dict):
        return [
            {"id": pattern_id, "template": pattern_template}
            for pattern_id, pattern_template in patterns.items()
        ]

    return patterns


def main_orchestrator(
    input_dir: str,
    output_dir: str,
    pattern_bank: dict,
    schema: list[str],
    id_to_string_map: dict,
    pattern_section: str = DEFAULT_PATTERN_SECTION,
) -> dict[str, int]:
    """Run augmentation for every PET pattern in the pattern bank."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    input_data = load_jsonl(Path(input_dir) / "train.jsonl")
    rows_by_pattern: dict[str, int] = {}

    for pattern in _iter_pattern_bank(pattern_bank, pattern_section):
        pattern_id = pattern["id"]
        pattern_output_path = output_path / f"{pattern_id}.jsonl"
        rows_by_pattern[pattern_id] = augment_data(
            input_data=input_data,
            pattern_id=pattern_id,
            pattern_template=pattern["template"],
            schema=schema,
            id_to_string_map=id_to_string_map,
            output_filepath=str(pattern_output_path),
        )

    return rows_by_pattern


def _default_output_dir(input_dir: Path) -> Path:
    return PROJECT_ROOT / DEFAULT_OUTPUT_BASE_DIR / "train" / f"{input_dir.name}_aug"


def main() -> None:
    seed = os.environ.get("AUG_RANDOM_SEED")
    if seed is not None:
        random.seed(int(seed))

    input_dir = Path(os.environ.get("AUG_INPUT_DIR", PROJECT_ROOT / DEFAULT_FEWSHOT_DIR))
    output_dir = Path(os.environ.get("AUG_OUTPUT_DIR", _default_output_dir(input_dir)))
    patterns_path = Path(os.environ.get("AUG_PATTERNS_PATH", SCRIPT_DIR / "patterns.yaml"))
    mapping_path = Path(os.environ.get("AUG_MAPPING_PATH", SCRIPT_DIR.parent / "mapping.yaml"))
    dataset_name = os.environ.get("AUG_DATASET", DEFAULT_DATASET)
    pattern_section = os.environ.get("AUG_PATTERN_SECTION", DEFAULT_PATTERN_SECTION)

    pattern_bank = load_yaml(patterns_path)
    dataset_mapping = load_yaml(mapping_path)[dataset_name]
    schema = dataset_mapping["entity_types"]
    id_to_string_map = dataset_mapping["id_to_label"]

    rows_by_pattern = main_orchestrator(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        pattern_bank=pattern_bank,
        schema=schema,
        id_to_string_map=id_to_string_map,
        pattern_section=pattern_section,
    )

    total_rows = sum(rows_by_pattern.values())
    print(f"Finished augmentation: {total_rows} rows across {len(rows_by_pattern)} pattern files.")


if __name__ == "__main__":
    main()
