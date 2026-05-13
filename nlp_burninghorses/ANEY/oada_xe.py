from __future__ import annotations

from collections import defaultdict
from itertools import permutations
import re

import torch


TARGET_ENTITY_RE = re.compile(
    r"\[([^\]]+)\]([A-Za-z][A-Za-z0-9_-]*)"
)


def parse_target_text(target_text: str) -> list[dict]:
    """
    Convert:

        [Briton]MISC [Mike Tyson]PER

    into:

        [
            {"text": "Briton", "type": "MISC"},
            {"text": "Mike Tyson", "type": "PER"},
        ]
    """
    entities = []

    for match in TARGET_ENTITY_RE.finditer(target_text):
        entities.append(
            {
                "text": match.group(1).strip(),
                "type": match.group(2).strip(),
            }
        )

    return entities


def format_entities_as_target(entities: list[dict]) -> str:
    """
    Convert entity dicts back into target_text format.
    """
    if not entities:
        return "None"

    return " ".join(
        f"[{ent['text']}]{ent['type']}"
        for ent in entities
    )


def make_same_type_permutation_targets(
    target_text: str,
) -> list[str]:
    """
    Generate equivalent target sequences where entities of the same type
    may appear in any order.

    Example:

        [Briton]MISC
        [World Boxing Council]ORG
        [WBC]ORG
        [Mike Tyson]PER

    becomes:

        [Briton]MISC [World Boxing Council]ORG [WBC]ORG [Mike Tyson]PER
        [Briton]MISC [WBC]ORG [World Boxing Council]ORG [Mike Tyson]PER
    """
    entities = parse_target_text(target_text)

    grouped = defaultdict(list)
    type_order = []

    for ent in entities:
        ent_type = ent["type"]

        if ent_type not in grouped:
            type_order.append(ent_type)

        grouped[ent_type].append(ent)

    per_type_options = []

    for ent_type in type_order:
        ents = grouped[ent_type]

        if len(ents) <= 1:
            per_type_options.append([ents])
        else:
            per_type_options.append(list(permutations(ents)))

    targets = []

    def backtrack(i: int, current: list[dict]) -> None:
        if i == len(per_type_options):
            targets.append(format_entities_as_target(current))
            return

        for option in per_type_options[i]:
            backtrack(i + 1, current + list(option))

    backtrack(0, [])

    return sorted(set(targets))


def seq2seq_ce_loss_for_targets(
    model,
    tokenizer,
    input_text: str,
    target_texts: list[str],
    device: torch.device,
    max_source_length: int = 128,
    max_target_length: int = 128,
) -> list[tuple[str, float]]:
    """
    Compute normal seq2seq cross entropy for each candidate target.
    Used for inspection/testing, not efficient batch training.
    """
    losses = []

    source = tokenizer(
        [input_text],
        max_length=max_source_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    for target_text in target_texts:
        target = tokenizer(
            text_target=[target_text],
            max_length=max_target_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        labels = target["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100

        with torch.no_grad():
            outputs = model(
                input_ids=source["input_ids"],
                attention_mask=source["attention_mask"],
                labels=labels,
            )

        losses.append((target_text, float(outputs.loss.item())))

    return losses


def oada_xe_loss_single_example(
    model,
    tokenizer,
    input_text: str,
    canonical_target_text: str,
    equivalent_target_texts: list[str],
    device: torch.device,
    tau: float,
    max_source_length: int = 128,
    max_target_length: int = 128,
) -> torch.Tensor:
    """
    Annealed OADA-XE:

        L = (1 - tau) * normal_XE + tau * min_equivalent_XE

    tau = 0.0 -> pure normal XE
    tau = 1.0 -> pure OADA-XE
    """
    all_targets = [canonical_target_text] + [
        t for t in equivalent_target_texts if t != canonical_target_text
    ]

    source = tokenizer(
        [input_text],
        max_length=max_source_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    losses = []

    for target_text in all_targets:
        target = tokenizer(
            text_target=[target_text],
            max_length=max_target_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        labels = target["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(
            input_ids=source["input_ids"],
            attention_mask=source["attention_mask"],
            labels=labels,
        )

        losses.append(outputs.loss)

    normal_xe = losses[0]
    oada_xe = torch.stack(losses).min()

    return (1.0 - tau) * normal_xe + tau * oada_xe


