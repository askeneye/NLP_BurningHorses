from __future__ import annotations

import importlib.util
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping

from datasets import DatasetDict, load_dataset

# Load the dataset
dataset_name = "conll2003" # 
raw_datasets = load_dataset(dataset_name, trust_remote_code=True)
# CONLL_HF_DATASET = "BramVanroy/conll2003"

K_VALUES = [5, 10, 20, 50]
SPLIT_SEEDS = [42, 142, 242]

TEXT_COLUMN = "tokens"
LABEL_COLUMN = "ner_tags"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_k_split_module() -> Any:
    path = Path(__file__).resolve().parent / "K-split.py"
    spec = importlib.util.spec_from_file_location("k_split_impl", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load K-split from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_conll2003() -> DatasetDict:
    ds = load_dataset("conll2003", trust_remote_code=True)
    return DatasetDict(
        train=ds["train"].select_columns([TEXT_COLUMN, LABEL_COLUMN]),
        validation=ds["validation"].select_columns([TEXT_COLUMN, LABEL_COLUMN]),
        test=ds["test"].select_columns([TEXT_COLUMN, LABEL_COLUMN]),
    )


def dataset_to_examples(dataset) -> List[Mapping]:
    return [
        {
            "tokens": row[TEXT_COLUMN],
            "ner_tags": row[LABEL_COLUMN],
        }
        for row in dataset
    ]


def write_jsonl(path: Path, rows: List[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_metadata(path: Path, metadata: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    ksplit = load_k_split_module()
    raw = load_conll2003()

    label_list = list(raw["train"].features[LABEL_COLUMN].feature.names)
    id_to_label = {i: label for i, label in enumerate(label_list)}

    train_pool = dataset_to_examples(raw["train"])
    validation = dataset_to_examples(raw["validation"])
    test = dataset_to_examples(raw["test"])

    out_root = repo_root() / "data" / "processed" / "conll2003_kshot"
    out_root.mkdir(parents=True, exist_ok=True)

    for k in K_VALUES:
        for run_index, seed in enumerate(SPLIT_SEEDS):
            print(f"Creating K={k}, seed={seed}")

            support, rest = ksplit.greedy_k_shot_support_split(
                train_pool,
                k,
                labels_key=LABEL_COLUMN,
                id2label=id_to_label,
                rng=random.Random(seed),
            )

            split_dir = out_root / f"k{k}_seed{seed}"

            write_jsonl(split_dir / "train.jsonl", support)
            write_jsonl(split_dir / "unlabeled_pool.jsonl", rest)
            write_jsonl(split_dir / "validation.jsonl", validation)
            write_jsonl(split_dir / "test.jsonl", test)

            metadata = {
                "dataset": dataset_name,
                "k": k,
                "seed": seed,
                "run_index": run_index,
                "num_train_support_sentences": len(support),
                "num_unlabeled_pool_sentences": len(rest),
                "num_validation_sentences": len(validation),
                "num_test_sentences": len(test),
                "label_list": label_list,
                "id_to_label": id_to_label,
                "text_column": TEXT_COLUMN,
                "label_column": LABEL_COLUMN,
            }

            write_metadata(split_dir / "metadata.json", metadata)

            print(
                f"  train={len(support)} "
                f"unlabeled_pool={len(rest)} "
                f"validation={len(validation)} "
                f"test={len(test)}"
            )

    print(f"\nDone. Splits written to: {out_root}")


if __name__ == "__main__":
    main()