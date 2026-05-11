"""
Greedy K-shot support sampling (Algorithm 1) from:

  Yang & Katiyar (2020). "Simple and Effective Few-Shot Named Entity Recognition
  with Structured Nearest Neighbor Learning." EMNLP.

Entity classes are sorted by increasing frequency in the pool; sentences are
sampled without replacement so that each class reaches K *entity spans*
(examples), updating counts for every entity type present in each added
sentence (so frequent types may exceed K).
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Union

# ---------------------------------------------------------------------------
# BIO / IO tag helpers
# ---------------------------------------------------------------------------


def _label_str(
    tag: Union[int, str],
    id2label: Optional[Mapping[int, str]] = None,
) -> str:
    if isinstance(tag, str):
        return tag
    if id2label is None:
        raise ValueError("Integer ner_tags require id2label (or label_names list).")
    return id2label[int(tag)]


def entity_span_counts_per_sentence(
    labels: Sequence[Union[int, str]],
    id2label: Optional[Mapping[int, str]] = None,
) -> Counter:
    """
    Count named-entity *spans* per coarse type for one sentence (BIO/IO).

    O and special labels outside BIO/IO are skipped. Span boundaries follow BIO:
    B-T starts a span; I-T continues only the same type T.
    """
    n = len(labels)
    if n == 0:
        return Counter()

    str_labels = [_label_str(t, id2label) for t in labels]
    counts: Counter = Counter()
    i = 0

    while i < n:
        lab = str_labels[i]
        if lab in ("O", "-100") or lab.startswith("[") or lab == "PAD":
            i += 1
            continue

        if lab.startswith("B-"):
            typ = lab[2:]
        elif lab.startswith("I-"):
            typ = lab[2:]
        else:
            i += 1
            continue

        j = i + 1
        while j < n:
            nxt = str_labels[j]
            if nxt == f"I-{typ}":
                j += 1
            else:
                break
        counts[typ] += 1
        i = j

    return counts


def corpus_entity_frequencies(
    examples: Sequence[Mapping],
    tokens_key: str = "tokens",
    labels_key: str = "ner_tags",
    id2label: Optional[Mapping[int, str]] = None,
) -> Counter:
    """Total entity-span counts per type over all examples."""
    total: Counter = Counter()
    for ex in examples:
        labs = ex[labels_key]
        total.update(entity_span_counts_per_sentence(labs, id2label=id2label))
    return total


def _types_in_sentence(
    labels: Sequence[Union[int, str]],
    id2label: Optional[Mapping[int, str]],
) -> Set[str]:
    return set(entity_span_counts_per_sentence(labels, id2label=id2label).keys())


# ---------------------------------------------------------------------------
# Algorithm 1: Greedy K-shot sampling
# ---------------------------------------------------------------------------


def greedy_k_shot_support_indices(
    examples: Sequence[Mapping],
    k: int,
    *,
    tokens_key: str = "tokens",
    labels_key: str = "ner_tags",
    id2label: Optional[Mapping[int, str]] = None,
    entity_types: Optional[Iterable[str]] = None,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """
    Return indices of examples selected into the support set S.

    Matches EMNLP 2020 Alg. 1: sort types by ascending frequency in X, then
    for each type in order, while its span count in S is < K, sample uniformly
    among remaining sentences that contain that type, add without replacement,
    and add span counts for all types in the chosen sentence to the tallies.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    rng = rng or random.Random()

    n = len(examples)
    pool_freq = corpus_entity_frequencies(
        examples, tokens_key=tokens_key, labels_key=labels_key, id2label=id2label
    )
    if entity_types is not None:
        classes = list(dict.fromkeys(entity_types))
    else:
        classes = list(pool_freq.keys())

    if not classes:
        return []

    classes.sort(key=lambda c: (pool_freq.get(c, 0), c))

    unused: Set[int] = set(range(n))
    support_indices: List[int] = []
    count_in_s: MutableMapping[str, int] = defaultdict(int)

    for ci in classes:
        while count_in_s[ci] < k:
            candidates = [
                idx
                for idx in unused
                if ci in _types_in_sentence(examples[idx][labels_key], id2label)
            ]
            if not candidates:
                break
            chosen = rng.choice(candidates)
            unused.remove(chosen)
            support_indices.append(chosen)
            spans = entity_span_counts_per_sentence(
                examples[chosen][labels_key], id2label=id2label
            )
            for t, c in spans.items():
                count_in_s[t] += c

    return support_indices


def greedy_k_shot_support_split(
    examples: Sequence[Mapping],
    k: int,
    *,
    tokens_key: str = "tokens",
    labels_key: str = "ner_tags",
    id2label: Optional[Mapping[int, str]] = None,
    entity_types: Optional[Iterable[str]] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Mapping], List[Mapping]]:
    """
    Split `examples` into (support, rest) using greedy K-shot sampling.

    Support contains the sentences selected by Algorithm 1; the remainder is
    everything else, order preserved within each part (support order follows
    selection order).
    """
    idx = greedy_k_shot_support_indices(
        examples,
        k,
        tokens_key=tokens_key,
        labels_key=labels_key,
        id2label=id2label,
        entity_types=entity_types,
        rng=rng,
    )
    taken = set(idx)
    support = [examples[i] for i in idx]
    rest = [examples[i] for i in range(len(examples)) if i not in taken]
    return support, rest


def load_conll_sentences(
    path: str,
    *,
    word_column: int = 0,
    tag_column: int = 1,
    delimiter: Optional[str] = None,
) -> List[Dict[str, List]]:
    """
    Read a CoNLL-style file (blank line between sentences) into
    [{"tokens": [...], "ner_tags": [...]}, ...] with string tags.
    Lines starting with # or empty lines separate sentences.
    """
    sentences: List[Dict[str, List]] = []
    tokens: List[str] = []
    tags: List[str] = []

    def flush() -> None:
        nonlocal tokens, tags
        if tokens:
            sentences.append({"tokens": list(tokens), "ner_tags": list(tags)})
        tokens, tags = [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                flush()
                continue
            parts = line.split(delimiter) if delimiter else line.split()
            if len(parts) <= max(word_column, tag_column):
                continue
            tokens.append(parts[word_column])
            tags.append(parts[tag_column])
        flush()

    return sentences
