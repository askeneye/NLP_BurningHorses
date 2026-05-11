from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path("data/interim/conll2003_kshot_bert")

VALIDATION_PATH = ROOT / "validation.jsonl"
MINI_VAL_PATH = ROOT / "mini_val.jsonl"

BACKUP_VALIDATION_PATH = ROOT / "validation_backup.jsonl"


def load_jsonl(path: Path) -> List[Dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_key(example: Dict) -> Tuple:
    """
    Creates a hashable unique key for comparison.

    We use both tokens and labels to avoid accidental collisions.
    """
    return (
        tuple(example["tokens"]),
        tuple(example["ner_tags"]),
    )


def main() -> None:
    validation = load_jsonl(VALIDATION_PATH)
    mini_eval = load_jsonl(MINI_VAL_PATH)

    print(f"Loaded validation: {len(validation)}")
    print(f"Loaded mini_eval: {len(mini_eval)}")

    # Backup original validation
    if not BACKUP_VALIDATION_PATH.exists():
        write_jsonl(BACKUP_VALIDATION_PATH, validation)
        print(f"Backup written to: {BACKUP_VALIDATION_PATH}")

    mini_keys = {make_key(x) for x in mini_eval}

    filtered_validation = [
        row
        for row in validation
        if make_key(row) not in mini_keys
    ]

    removed = len(validation) - len(filtered_validation)

    write_jsonl(VALIDATION_PATH, filtered_validation)

    print(f"Removed {removed} overlapping sentences")
    print(f"New validation size: {len(filtered_validation)}")


if __name__ == "__main__":
    main()