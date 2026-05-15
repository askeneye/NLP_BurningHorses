from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path("data/interim/fewnerd_supervised_fine_bio_kshot_bert")

VALIDATION_PATH = ROOT / "validation.jsonl"
MINI_VAL_PATH = ROOT / "mini_val.jsonl"
METADATA_PATH = ROOT / "metadata.json"

BACKUP_VALIDATION_PATH = ROOT / "validation_before_resize_minival.jsonl"
BACKUP_MINI_VAL_PATH = ROOT / "mini_val_before_resize_minival.jsonl"

TARGET_MINI_VAL_SIZE = 500
SEED = 42


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def make_key(example: Dict) -> Tuple:
    return (
        tuple(example["tokens"]),
        tuple(example["ner_tags"]),
    )


def deduplicate(rows: List[Dict]) -> List[Dict]:
    seen = set()
    unique_rows = []

    for row in rows:
        key = make_key(row)
        if key not in seen:
            unique_rows.append(row)
            seen.add(key)

    return unique_rows


def main() -> None:
    validation = load_jsonl(VALIDATION_PATH)
    mini_val = load_jsonl(MINI_VAL_PATH)

    print(f"Loaded validation: {len(validation)}")
    print(f"Loaded mini_val: {len(mini_val)}")

    if not BACKUP_VALIDATION_PATH.exists():
        write_jsonl(BACKUP_VALIDATION_PATH, validation)
        print(f"Backup written to: {BACKUP_VALIDATION_PATH}")

    if not BACKUP_MINI_VAL_PATH.exists():
        write_jsonl(BACKUP_MINI_VAL_PATH, mini_val)
        print(f"Backup written to: {BACKUP_MINI_VAL_PATH}")

    full_validation_pool = deduplicate(validation + mini_val)

    if TARGET_MINI_VAL_SIZE > len(full_validation_pool):
        raise ValueError(
            f"TARGET_MINI_VAL_SIZE={TARGET_MINI_VAL_SIZE} is larger than "
            f"available pool size={len(full_validation_pool)}"
        )

    rng = random.Random(SEED)
    new_mini_val = rng.sample(full_validation_pool, k=TARGET_MINI_VAL_SIZE)

    new_mini_val_keys = {make_key(row) for row in new_mini_val}

    new_validation = [
        row
        for row in full_validation_pool
        if make_key(row) not in new_mini_val_keys
    ]

    write_jsonl(MINI_VAL_PATH, new_mini_val)
    write_jsonl(VALIDATION_PATH, new_validation)

    metadata = load_json(METADATA_PATH)
    metadata["num_raw_validation_sentences"] = len(full_validation_pool)
    metadata["num_validation_sentences"] = len(new_validation)
    metadata["num_mini_eval_sentences"] = len(new_mini_val)
    metadata["mini_eval_requested_size"] = TARGET_MINI_VAL_SIZE
    metadata["mini_eval_seed"] = SEED
    metadata["validation_excludes_mini_eval"] = True
    write_json(METADATA_PATH, metadata)

    overlap = {make_key(row) for row in new_validation} & {
        make_key(row) for row in new_mini_val
    }

    print("\nDone.")
    print(f"Full validation pool: {len(full_validation_pool)}")
    print(f"New mini_val size: {len(new_mini_val)}")
    print(f"New validation size: {len(new_validation)}")
    print(f"Overlap between validation and mini_val: {len(overlap)}")


if __name__ == "__main__":
    main()