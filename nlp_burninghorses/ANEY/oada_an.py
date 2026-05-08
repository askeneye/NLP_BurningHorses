from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class Entity:
    text: str
    type: str
    start: int  # inclusive token index
    end: int    # inclusive token index


def label_to_str(
    label: Union[int, str],
    id_to_label: Optional[Mapping[int, str]] = None,
) -> str:
    if isinstance(label, str):
        return label
    if id_to_label is None:
        raise ValueError("Integer labels require id_to_label.")
    return id_to_label[int(label)]


def extract_entities_from_bio(
    tokens: Sequence[str],
    labels: Sequence[Union[int, str]],
    id_to_label: Optional[Mapping[int, str]] = None,
) -> List[Entity]:
    """
    Convert BIO/IO tags into entity spans.

    Example:
        tokens = ["CNN", "'s", "David", "Ensor"]
        labels = ["B-ORG", "O", "B-PER", "I-PER"]

        -> [
            Entity("CNN", "ORG", 0, 0),
            Entity("David Ensor", "PER", 2, 3)
        ]
    """
    if len(tokens) != len(labels):
        raise ValueError("tokens and labels must have the same length")

    str_labels = [label_to_str(label, id_to_label) for label in labels]
    entities: List[Entity] = []

    i = 0
    while i < len(tokens):
        tag = str_labels[i]

        if tag in {"O", "-100", "PAD"} or tag.startswith("["):
            i += 1
            continue

        if tag.startswith("B-"):
            ent_type = tag[2:]
        elif tag.startswith("I-"):
            # Robust handling: treat stray I-X as a new span.
            ent_type = tag[2:]
        else:
            i += 1
            continue

        start = i
        j = i + 1

        while j < len(tokens) and str_labels[j] == f"I-{ent_type}":
            j += 1

        end = j - 1
        text = " ".join(tokens[start : end + 1])

        entities.append(Entity(text=text, type=ent_type, start=start, end=end))
        i = j

    return entities


def entity_types_from_label_list(label_list: Sequence[str]) -> List[str]:
    """
    Extract coarse entity types from BIO labels.

    Example:
        ["O", "B-PER", "I-PER", "B-ORG"] -> ["ORG", "PER"]
    """
    return sorted(
        {
            label[2:]
            for label in label_list
            if label != "O" and "-" in label
        }
    )


def generate_ordering_instructions(
    entity_types: Sequence[str],
) -> List[Tuple[str, ...]]:
    """
    Generate all type-order permutations.

    CoNLL-2003:
        ["PER", "LOC", "ORG", "MISC"] -> 24 permutations
    """
    return list(permutations(entity_types))


def group_entities_by_order(
    entities: Sequence[Entity],
    order: Sequence[str],
) -> List[Entity]:
    """
    Rearrange entities according to an entity-type ordering instruction.

    Within the same type, keep original sentence order for now.
    OADA-XE can later relax same-type ordering.
    """
    ordered_entities: List[Entity] = []

    for ent_type in order:
        same_type = [ent for ent in entities if ent.type == ent_type]
        same_type.sort(key=lambda ent: (ent.start, ent.end))
        ordered_entities.extend(same_type)

    return ordered_entities


def format_oada_input(
    tokens: Sequence[str],
    order: Sequence[str],
) -> str:
    sentence = " ".join(tokens)
    instruction = " ".join(order)
    return f"Order: {instruction} Sentence: {sentence}"


def format_oada_target(
    entities: Sequence[Entity],
    empty_target: str = "None",
) -> str:
    """
    Format target entity sequence.

    Example:
        David Ensor is PER ; CNN is ORG
    """
    if not entities:
        return empty_target

    return " ; ".join(f"{ent.text} is {ent.type}" for ent in entities)


def make_oada_pairs(
    example: Mapping,
    entity_types: Sequence[str],
    *,
    tokens_key: str = "tokens",
    labels_key: str = "ner_tags",
    id_to_label: Optional[Mapping[int, str]] = None,
    include_empty: bool = False,
) -> List[Dict]:
    """
    Create OADA input-output pairs for one BIO-tagged NER example.
    """
    tokens = example[tokens_key]
    labels = example[labels_key]

    entities = extract_entities_from_bio(tokens, labels, id_to_label=id_to_label)

    if not entities and not include_empty:
        return []

    rows: List[Dict] = []
    seen = set()

    for order in generate_ordering_instructions(entity_types):
        ordered_entities = group_entities_by_order(entities, order)
        target = format_oada_target(ordered_entities)

        key = tuple((ent.text, ent.type, ent.start, ent.end) for ent in ordered_entities)
        if key in seen:
            continue
        seen.add(key)

        rows.append(
            {
                "input": format_oada_input(tokens, order),
                "target": target,
                "order": list(order),
                "entities": [asdict(ent) for ent in ordered_entities],
                "tokens": list(tokens),
                "ner_tags": list(labels),
            }
        )

    return rows