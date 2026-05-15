from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping

from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DATASET_NAME = "DFKI-SLT/few-nerd"
FEWNERD_CONFIG = "supervised"

K_VALUES = [5, 10, 20, 50]
SPLIT_SEEDS = [42, 142, 242]

TEXT_COLUMN = "tokens"

# Original Few-NERD fine-grained IO labels
ORIGINAL_LABEL_COLUMN = "fine_ner_tags"

# Output label column after BIO conversion so downstream BERT code looks like CoNLL.
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


def make_bio_label_maps(
    original_label_list: List[str],
) -> tuple[List[str], Dict[str, int], Dict[int, str]]:
    bio_label_list = ["O"]

    for label in original_label_list:
        if label == "O":
            continue
        bio_label_list.append(f"B-{label}")
        bio_label_list.append(f"I-{label}")

    label_to_id = {label: i for i, label in enumerate(bio_label_list)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    return bio_label_list, label_to_id, id_to_label


def io_to_bio_ids(
    io_label_ids: List[int],
    io_id_to_label: Dict[int, str],
    bio_label_to_id: Dict[str, int],
) -> List[int]:
    bio_ids = []
    prev_label = "O"

    for label_id in io_label_ids:
        label = io_id_to_label[int(label_id)]

        if label == "O":
            bio_ids.append(bio_label_to_id["O"])
            prev_label = "O"
            continue

        prefix = "I" if label == prev_label else "B"
        bio_ids.append(bio_label_to_id[f"{prefix}-{label}"])
        prev_label = label

    return bio_ids


def load_fewnerd_bio() -> tuple[DatasetDict, List[str], Dict[int, str]]:
    
    ds = load_dataset(
        DATASET_NAME,
        FEWNERD_CONFIG,
        trust_remote_code=True,
    )

    original_label_list = list(
        ds["train"].features[ORIGINAL_LABEL_COLUMN].feature.names
    )
    io_id_to_label = {i: label for i, label in enumerate(original_label_list)}

    bio_label_list, bio_label_to_id, bio_id_to_label = make_bio_label_maps(
        original_label_list
    )

    output_features = Features(
        {
            TEXT_COLUMN: Sequence(Value("string")),
            LABEL_COLUMN: Sequence(Value("int64")),
        }
    )

    def convert_split(split) -> Dataset:
        rows = []

        for example in split:
            rows.append(
                {
                    TEXT_COLUMN: example[TEXT_COLUMN],
                    LABEL_COLUMN: io_to_bio_ids(
                        example[ORIGINAL_LABEL_COLUMN],
                        io_id_to_label,
                        bio_label_to_id,
                    ),
                }
            )

        return Dataset.from_list(rows, features=output_features)

    bio_ds = DatasetDict(
        train=convert_split(ds["train"]),
        validation=convert_split(ds["validation"]),
        test=convert_split(ds["test"]),
    )

    return bio_ds, bio_label_list, bio_id_to_label


def dataset_to_examples(dataset: Dataset) -> List[Mapping]:
    return [
        {
            TEXT_COLUMN: row[TEXT_COLUMN],
            LABEL_COLUMN: row[LABEL_COLUMN],
        }
        for row in dataset
    ]


def strip_labels(rows: List[Mapping]) -> List[Mapping]:
    return [{TEXT_COLUMN: row[TEXT_COLUMN]} for row in rows]


def split_mini_val(
    validation_rows: List[Mapping],
    mini_val_size: int,
    seed: int,
) -> tuple[List[Mapping], List[Mapping]]:
    rng = random.Random(seed)
    mini_val = rng.sample(validation_rows, k=min(mini_val_size, len(validation_rows)))

    mini_eval_keys = {
        (
            tuple(example[TEXT_COLUMN]),
            tuple(example[LABEL_COLUMN]),
        )
        for example in mini_val
    }

    filtered_validation = [
        example
        for example in validation_rows
        if (
            tuple(example[TEXT_COLUMN]),
            tuple(example[LABEL_COLUMN]),
        )
        not in mini_eval_keys
    ]

    return mini_val, filtered_validation


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
    raw, label_list, id_to_label = load_fewnerd_bio()

    train_pool = dataset_to_examples(raw["train"])
    validation = dataset_to_examples(raw["validation"])
    test = dataset_to_examples(raw["test"])

    out_root = (
        repo_root()
        / "data"
        / "interim"
        / "fewnerd_supervised_fine_bio_kshot_bert"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    # Shared top-level splits
    mini_val_seed = 42
    mini_val_size = 200

    mini_val, filtered_validation = split_mini_val(
        validation,
        mini_val_size,
        mini_val_seed,
    )

    write_jsonl(out_root / "validation.jsonl", filtered_validation)
    write_jsonl(out_root / "test.jsonl", test)
    write_jsonl(out_root / "mini_val.jsonl", mini_val)

    write_metadata(
        out_root / "metadata.json",
        {
            "dataset": DATASET_NAME,
            "config": FEWNERD_CONFIG,
            "label_level": "fine",
            "original_label_column": ORIGINAL_LABEL_COLUMN,
            "label_column": LABEL_COLUMN,
            "original_tagging_scheme": "IO",
            "tagging_scheme": "BIO",
            "num_raw_validation_sentences": len(validation),
            "num_validation_sentences": len(filtered_validation),
            "num_test_sentences": len(test),
            "num_mini_val_sentences": len(mini_val),
            "mini_val_source": "validation",
            "mini_val_seed": mini_val_seed,
            "mini_val_requested_size": mini_val_size,
            "validation_excludes_mini_val": True,
            "label_list": label_list,
            "id_to_label": id_to_label,
            "text_column": TEXT_COLUMN,
            "label_column": LABEL_COLUMN,
        },
    )

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
            write_jsonl(split_dir / "unlabeled_pool.jsonl", strip_labels(rest))

            metadata = {
                "dataset": DATASET_NAME,
                "config": FEWNERD_CONFIG,
                "label_level": "fine",
                "original_label_column": ORIGINAL_LABEL_COLUMN,
                "label_column": LABEL_COLUMN,
                "original_tagging_scheme": "IO",
                "tagging_scheme": "BIO",
                "k": k,
                "seed": seed,
                "run_index": run_index,
                "num_train_support_sentences": len(support),
                "num_unlabeled_pool_sentences": len(rest),
                "unlabeled_pool_contains_gold_labels": False,
                "shared_validation_path": "../validation.jsonl",
                "shared_test_path": "../test.jsonl",
                "shared_mini_val_path": "../mini_val.jsonl",
                "label_list": label_list,
                "id_to_label": id_to_label,
                "text_column": TEXT_COLUMN,
                "label_column": LABEL_COLUMN,
            }

            write_metadata(split_dir / "metadata.json", metadata)

            print(
                f"  train={len(support)} "
                f"unlabeled_pool={len(rest)}"
            )

    print(f"\nSplits written to: {out_root}")


if __name__ == "__main__":
    main()