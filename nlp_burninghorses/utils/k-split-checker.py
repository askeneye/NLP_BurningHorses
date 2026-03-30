"""
Sanity-check greedy K-shot support sampling on NER train data (English EWT IOB2 or CoNLL-2003).

Run from repo root, examples:

  python nlp_burninghorses/utils/k-split-checker.py
  python nlp_burninghorses/utils/k-split-checker.py --dataset conll -k 10 20
  python nlp_burninghorses/utils/k-split-checker.py --dataset conll --conll-train data/raw/conll2003/eng.train
  python nlp_burninghorses/utils/k-split-checker.py --dataset ewt --k-shots 5 --seed 0
"""

from __future__ import annotations

import argparse
import importlib.util
import random
from collections import Counter
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple


def _nlp_burninghorses_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_k_split_module() -> Any:
    path = _nlp_burninghorses_dir() / "K-split.py"
    spec = importlib.util.spec_from_file_location("k_split_impl", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load K-split from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_en_ewt_train_examples(repo_root: Optional[Path] = None) -> List[Mapping[str, List]]:
    """
    Load `data/raw/en_ewt-ud-train.iob2` as a list of
    {"tokens": [...], "ner_tags": [...]} with string IOB2 tags.
    """
    ksplit = _load_k_split_module()
    root = repo_root if repo_root is not None else _repo_root()
    train_path = root / "data" / "raw" / "en_ewt-ud-train.iob2"
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    return ksplit.load_conll_sentences(
        str(train_path),
        delimiter="\t",
        word_column=1,
        tag_column=2,
    )


CONLL_HF_DATASET = "BramVanroy/conll2003"


def load_conll_train_from_file(path: Path) -> List[Mapping[str, List]]:
    """
    Parse a classic CoNLL-style file: non-empty lines are whitespace fields, token = first
    column, NER tag = last column (CoNLL-2003 English); blank line ends a sentence.
    """
    sentences: List[Mapping[str, List]] = []
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
            if not line.strip():
                flush()
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            tokens.append(parts[0])
            tags.append(parts[-1])
        flush()

    if not sentences:
        raise ValueError(f"No sentences parsed from {path}")
    return sentences


def load_conll_train_examples(
    conll_train_path: Optional[Path] = None,
) -> List[Mapping[str, List]]:
    """
    CoNLL-2003 train as {"tokens": [...], "ner_tags": [...]} with string BIO tags.

    If ``conll_train_path`` is set, reads that file (e.g. official ``eng.train``).
    Otherwise loads ``BramVanroy/conll2003`` from the Hub (Parquet; works with recent
    ``datasets``, same as ``BERT_Baseline_conll_parquet.py``).
    """
    if conll_train_path is not None:
        p = Path(conll_train_path)
        if not p.is_file():
            raise FileNotFoundError(p)
        return load_conll_train_from_file(p)

    from datasets import load_dataset

    ds = load_dataset(CONLL_HF_DATASET)
    label_names = ds["train"].features["ner_tags"].feature.names
    examples: List[Mapping[str, List]] = []
    n = len(ds["train"])
    for i in range(n):
        row = ds["train"][i]
        toks = row["tokens"]
        ner = row["ner_tags"]
        if ner and isinstance(ner[0], str):
            tags = list(ner)
        else:
            tags = [label_names[int(j)] for j in ner]
        examples.append({"tokens": list(toks), "ner_tags": tags})
    return examples


def aggregate_entity_span_counts(
    examples: Sequence[Mapping],
    *,
    ksplit: Any,
    labels_key: str = "ner_tags",
    id2label: Optional[Mapping[int, str]] = None,
) -> Counter:
    total: Counter = Counter()
    for ex in examples:
        total.update(
            ksplit.entity_span_counts_per_sentence(ex[labels_key], id2label=id2label)
        )
    return total


def _sorted_types_union(*counters: Counter) -> List[str]:
    keys = set()
    for c in counters:
        keys.update(c.keys())
    return sorted(keys)


def _print_count_report(
    title: str,
    pool: Counter,
    support: Counter,
    k: int,
    types: Sequence[str],
) -> None:
    print(title)
    w = max(len("Entity type"), max((len(t) for t in types), default=0))
    header = f"{'Entity type':<{w}}  {'Pool':>8}  {'Support':>8}  {'>=K':>5}"
    print(header)
    print("-" * len(header))
    for t in types:
        p = pool.get(t, 0)
        s = support.get(t, 0)
        ok = "yes" if s >= k else "no"
        print(f"{t:<{w}}  {p:8d}  {s:8d}  {ok:>5}")
    print()


def run_k_split_sanity(
    examples: List[Mapping],
    dataset_title: str,
    k: int = 5,
    seed: int = 42,
) -> Tuple[List[Mapping], List[Mapping], Counter, Counter]:
    """
    Run greedy K-shot split on `examples` and print pool vs support span counts.

    Returns (support_examples, rest_examples, pool_span_counts, support_span_counts).
    """
    ksplit = _load_k_split_module()
    rng = random.Random(seed)

    pool_counts = aggregate_entity_span_counts(examples, ksplit=ksplit)
    support, rest = ksplit.greedy_k_shot_support_split(
        examples, k, labels_key="ner_tags", id2label=None, rng=rng
    )
    support_counts = aggregate_entity_span_counts(support, ksplit=ksplit)

    rarest_first = sorted(pool_counts.keys(), key=lambda c: (pool_counts[c], c))

    print(f"{dataset_title} — greedy K-shot support sanity check")
    print("  (Yang & Katiyar 2020, Algorithm 1; counts = entity spans)")
    print()
    print(f"  Train sentences: {len(examples)}")
    print(f"  K (target spans per type): {k}")
    print(f"  RNG seed: {seed}")
    print(f"  Support sentences: {len(support)}")
    print(f"  Remaining (not in support): {len(rest)}")
    print()
    print("  Entity types sorted rarest-first (algorithm order):")
    print("   ", ", ".join(rarest_first))
    print()

    types = _sorted_types_union(pool_counts, support_counts)
    _print_count_report("  Span counts (pool vs support):", pool_counts, support_counts, k, types)

    shortfall = [t for t in pool_counts if support_counts.get(t, 0) < k]
    if shortfall:
        print("  Types that did not reach K spans in support (pool exhausted):")
        for t in sorted(shortfall, key=lambda x: (pool_counts[x], x)):
            print(f"    {t}: need {k}, have {support_counts.get(t, 0)} (pool had {pool_counts[t]})")
        print()

    return support, rest, pool_counts, support_counts


def run_en_ewt_k_split_sanity(
    k: int = 5,
    seed: int = 42,
    *,
    repo_root: Optional[Path] = None,
) -> Tuple[List[Mapping], List[Mapping], Counter, Counter]:
    examples = load_en_ewt_train_examples(repo_root)
    return run_k_split_sanity(
        examples,
        "English EWT Universal NER",
        k=k,
        seed=seed,
    )


def run_conll_k_split_sanity(
    k: int = 5,
    seed: int = 42,
    *,
    conll_train_path: Optional[Path] = None,
) -> Tuple[List[Mapping], List[Mapping], Counter, Counter]:
    examples = load_conll_train_examples(conll_train_path=conll_train_path)
    src = f"CoNLL-2003 ({conll_train_path})" if conll_train_path else f"CoNLL-2003 ({CONLL_HF_DATASET})"
    return run_k_split_sanity(
        examples,
        src,
        k=k,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedy K-shot support sampling sanity check on NER train data."
    )
    parser.add_argument(
        "--dataset",
        choices=("ewt", "conll"),
        default="ewt",
        help=(
            "Train source: ewt = local en_ewt-ud-train.iob2; "
            f"conll = {CONLL_HF_DATASET} train (or --conll-train file)."
        ),
    )
    parser.add_argument(
        "--conll-train",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Optional path to a CoNLL-formatted train file (token = 1st column, NER = last column). "
            "If omitted with --dataset conll, loads from the Hub (requires network/cache)."
        ),
    )
    parser.add_argument(
        "-k",
        "--k-shots",
        type=int,
        nargs="+",
        default=[5],
        metavar="K",
        help="Target entity spans per type; pass several integers (e.g. -k 10 20). Default: 5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed passed to random.Random for sampling. Default: 42.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing data/raw/ (EWT only; default: inferred from this file).",
    )
    args = parser.parse_args()
    for i, k in enumerate(args.k_shots):
        if k < 1:
            parser.error(f"-k/--k-shots must be >= 1, got {k}")
        if i:
            print()
            print("=" * 70)
            print(f" K = {k} ")
            print("=" * 70)
        if args.dataset == "ewt":
            run_en_ewt_k_split_sanity(k=k, seed=args.seed, repo_root=args.repo_root)
        else:
            run_conll_k_split_sanity(
                k=k,
                seed=args.seed,
                conll_train_path=args.conll_train,
            )


if __name__ == "__main__":
    main()
