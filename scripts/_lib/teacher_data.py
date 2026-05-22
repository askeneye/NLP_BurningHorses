from __future__ import annotations

import itertools
from typing import Any


ENTITY_TYPE_WORDS = {
    "PER": "person",
    "ORG": "organization",
    "LOC": "location",
    "MISC": "other",
}


def verbalize_entity_type(entity_type: str) -> str:
    return ENTITY_TYPE_WORDS.get(entity_type, entity_type.lower())


def detokenize_tokens(tokens: list[str]) -> str:
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


def extract_entities_by_type(tokens: list[str], ner_tags: list[int], id_to_label: dict[int, str]) -> dict[str, list[str]]:
    entities_by_type: dict[str, list[str]] = {}
    current_type: str | None = None
    current_tokens: list[str] = []

    def flush() -> None:
        nonlocal current_type, current_tokens
        if current_type and current_tokens:
            entities_by_type.setdefault(current_type, []).append(detokenize_tokens(current_tokens))
        current_type = None
        current_tokens = []

    for token, tag_id in zip(tokens, ner_tags):
        tag = id_to_label[int(tag_id)]
        if tag == "O" or "-" not in tag:
            flush()
            continue
        prefix, entity_type = tag.split("-", 1)
        if prefix == "B" or current_type != entity_type:
            flush()
            current_type = entity_type
            current_tokens = [token]
        elif prefix == "I":
            current_tokens.append(token)
        else:
            flush()
    flush()
    return entities_by_type


def extract_entities_left_to_right(tokens: list[str], ner_tags: list[int], id_to_label: dict[int, str]) -> list[tuple[str, str]]:
    entities: list[tuple[str, str]] = []
    current_type: str | None = None
    current_tokens: list[str] = []

    def flush() -> None:
        nonlocal current_type, current_tokens
        if current_type and current_tokens:
            entities.append((detokenize_tokens(current_tokens), current_type))
        current_type = None
        current_tokens = []

    for token, tag_id in zip(tokens, ner_tags):
        tag = id_to_label[int(tag_id)]
        if tag == "O" or "-" not in tag:
            flush()
            continue
        prefix, entity_type = tag.split("-", 1)
        if prefix == "B" or current_type != entity_type:
            flush()
            current_type = entity_type
            current_tokens = [token]
        elif prefix == "I":
            current_tokens.append(token)
        else:
            flush()
    flush()
    return entities


def generate_target_by_order(entities_by_type: dict[str, list[str]], order: tuple[str, ...]) -> str:
    parts: list[str] = []
    for entity_type in order:
        for entity_text in entities_by_type.get(entity_type, []):
            parts.append(f"[{entity_text}] {verbalize_entity_type(entity_type)}")
    return " ".join(parts)


def generate_target_left_to_right(entities: list[tuple[str, str]]) -> str:
    return " ".join(f"[{entity_text}] {verbalize_entity_type(entity_type)}" for entity_text, entity_type in entities)


def apply_pattern(sentence_text: str, order_text: str, pattern_template: str) -> str:
    return pattern_template.format(
        SEN=sentence_text,
        PERM=order_text,
        sentence_text=sentence_text,
        order=order_text,
    )


def canonical_pattern_id(index: int) -> str:
    return f"pattern_{index:02d}"


def entity_schema_from_label_list(label_list: list[str]) -> list[str]:
    schema: list[str] = []
    for label in label_list:
        if label.startswith("B-"):
            entity_type = label[2:]
            if entity_type not in schema:
                schema.append(entity_type)
    return schema


def verbalize_entity_order(order: tuple[str, ...]) -> str:
    return ", ".join(verbalize_entity_type(entity_type) for entity_type in order)


def oada_permutations(entity_schema: list[str]) -> list[tuple[str, ...]]:
    if len(entity_schema) <= 4:
        return list(itertools.permutations(entity_schema))
    # Not used in this promoted CoNLL run, but keep deterministic fallback.
    return list(itertools.permutations(entity_schema))[:20]


def normalize_pattern_bank(pattern_config: dict[str, Any], section: str = "patterns") -> list[dict[str, str]]:
    raw_patterns = pattern_config.get(section, [])
    if isinstance(raw_patterns, dict):
        return [{"id": key, "template": value} for key, value in raw_patterns.items()]
    return list(raw_patterns)
