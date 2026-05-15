from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

ROOT = Path(
    "data/interim/fewnerd_supervised_fine_bio_kshot_bert"
)

LABEL_COLUMN = "ner_tags"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


def entity_type_from_bio(label: str) -> str | None:
    if label == "O":
        return None

    return label[2:]  # remove B- / I-


def count_entity_mentions(
    rows: List[Mapping],
    id_to_label: Dict[int, str],
) -> Counter:
    counts = Counter()

    for row in rows:
        labels = row[LABEL_COLUMN]

        for label_id in labels:
            label = id_to_label[str(label_id)]

            # Only count B- tags to count entities once
            if label.startswith("B-"):
                entity_type = entity_type_from_bio(label)
                counts[entity_type] += 1

    return counts


def find_split_dirs(root: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in root.iterdir()
            if p.is_dir() and p.name.startswith("k")
        ],
        key=lambda p: p.name,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    top_metadata = load_json(ROOT / "metadata.json")
    id_to_label = top_metadata["id_to_label"]

    split_dirs = find_split_dirs(ROOT)

    print(f"Found {len(split_dirs)} split folders\n")

    for split_dir in split_dirs:
        metadata = load_json(split_dir / "metadata.json")

        k = metadata["k"]
        seed = metadata["seed"]

        train_rows = load_jsonl(split_dir / "train.jsonl")

        counts = count_entity_mentions(
            train_rows,
            id_to_label,
        )

        num_entity_types = len(counts)

        min_mentions = min(counts.values()) if counts else 0
        max_mentions = max(counts.values()) if counts else 0
        avg_mentions = (
            sum(counts.values()) / len(counts)
            if counts
            else 0.0
        )

        missing_types = []

        all_entity_types = {
            label[2:]
            for label in id_to_label.values()
            if label.startswith("B-")
        }

        for entity_type in sorted(all_entity_types):
            if entity_type not in counts:
                missing_types.append(entity_type)

        print("=" * 70)
        print(f"{split_dir.name}")
        print(f"K={k} | seed={seed}")
        print(f"train sentences: {len(train_rows)}")
        print(f"covered entity types: {num_entity_types}")
        print(f"min mentions per type: {min_mentions}")
        print(f"max mentions per type: {max_mentions}")
        print(f"avg mentions per type: {avg_mentions:.2f}")

        if missing_types:
            print(f"missing entity types ({len(missing_types)}):")
            for t in missing_types:
                print(f"  - {t}")
        else:
            print("all entity types covered")

        print()


if __name__ == "__main__":
    main()