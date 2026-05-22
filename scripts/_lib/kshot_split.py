from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Mapping


def entity_span_counts_per_sentence(label_ids: list[int], id_to_label: dict[int, str]) -> Counter[str]:
    """Count BIO spans per coarse type in one sentence."""
    labels = [id_to_label[int(label_id)] for label_id in label_ids]
    counts: Counter[str] = Counter()
    index = 0
    while index < len(labels):
        label = labels[index]
        if label == "O":
            index += 1
            continue
        if label.startswith("B-"):
            entity_type = label[2:]
        elif label.startswith("I-"):
            entity_type = label[2:]
        else:
            index += 1
            continue

        next_index = index + 1
        while next_index < len(labels) and labels[next_index] == f"I-{entity_type}":
            next_index += 1
        counts[entity_type] += 1
        index = next_index
    return counts


def corpus_entity_frequencies(examples: list[dict], id_to_label: dict[int, str]) -> Counter[str]:
    total: Counter[str] = Counter()
    for example in examples:
        total.update(entity_span_counts_per_sentence(example["ner_tags"], id_to_label))
    return total


def _types_in_sentence(example: Mapping, id_to_label: dict[int, str]) -> set[str]:
    return set(entity_span_counts_per_sentence(list(example["ner_tags"]), id_to_label).keys())


def greedy_k_shot_support_split(
    examples: list[dict],
    k: int,
    *,
    id_to_label: dict[int, str],
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """
    Greedy K-shot support split following Yang & Katiyar (2020) algorithm.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    pool_freq = corpus_entity_frequencies(examples, id_to_label)
    classes = sorted(pool_freq.keys(), key=lambda class_name: (pool_freq[class_name], class_name))

    unused = set(range(len(examples)))
    support_indices: list[int] = []
    count_in_support: dict[str, int] = defaultdict(int)

    for class_name in classes:
        while count_in_support[class_name] < k:
            candidates = [index for index in unused if class_name in _types_in_sentence(examples[index], id_to_label)]
            if not candidates:
                break
            chosen = rng.choice(candidates)
            unused.remove(chosen)
            support_indices.append(chosen)
            spans = entity_span_counts_per_sentence(examples[chosen]["ner_tags"], id_to_label)
            for entity_type, count in spans.items():
                count_in_support[entity_type] += count

    support_set = set(support_indices)
    support = [examples[index] for index in support_indices]
    rest = [examples[index] for index in range(len(examples)) if index not in support_set]
    return support, rest
