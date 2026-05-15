from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping


DATA_ROOT = Path("data/interim/conll2003_kshot_bert")
RESULTS_ROOT = Path("results/Training_data_entity_distributions")

LABEL_COLUMN = "ner_tags"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def find_split_dirs(root: Path) -> List[Path]:
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("k")],
        key=lambda p: p.name,
    )


def count_entity_mentions(
    rows: List[Mapping],
    id_to_label: Dict[str, str],
) -> Counter:
    counts = Counter()

    for row in rows:
        for label_id in row[LABEL_COLUMN]:
            label = id_to_label[str(label_id)]

            if label.startswith("B-"):
                entity_type = label[2:]
                counts[entity_type] += 1

    return counts


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to write.")

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    top_metadata = load_json(DATA_ROOT / "metadata.json")
    id_to_label = top_metadata["id_to_label"]

    entity_types = sorted(
        {
            label[2:]
            for label in id_to_label.values()
            if label.startswith("B-")
        }
    )

    rows = []
    summary_rows = []

    for split_dir in find_split_dirs(DATA_ROOT):
        metadata = load_json(split_dir / "metadata.json")

        k = int(metadata["k"])
        seed = int(metadata["seed"])
        split_name = split_dir.name

        train_rows = load_jsonl(split_dir / "train.jsonl")
        counts = count_entity_mentions(train_rows, id_to_label)

        total_mentions = sum(counts.values())

        for entity_type in entity_types:
            mention_count = counts.get(entity_type, 0)

            rows.append(
                {
                    "dataset": "conll2003",
                    "config": "standard",
                    "label_level": "coarse",
                    "tagging_scheme": "BIO",
                    "split_name": split_name,
                    "k": k,
                    "seed": seed,
                    "entity_type": entity_type,
                    "mention_count": mention_count,
                    "target_k": k,
                    "meets_k": mention_count >= k,
                    "train_sentences": len(train_rows),
                    "total_entity_mentions": total_mentions,
                    "share_of_mentions": (
                        mention_count / total_mentions
                        if total_mentions > 0
                        else 0.0
                    ),
                }
            )

        summary_rows.append(
            {
                "dataset": "conll2003",
                "config": "standard",
                "label_level": "coarse",
                "tagging_scheme": "BIO",
                "split_name": split_name,
                "k": k,
                "seed": seed,
                "train_sentences": len(train_rows),
                "covered_entity_types": sum(
                    1 for x in entity_types if counts.get(x, 0) > 0
                ),
                "num_entity_types": len(entity_types),
                "min_mentions": min(counts.get(x, 0) for x in entity_types),
                "max_mentions": max(counts.get(x, 0) for x in entity_types),
                "total_entity_mentions": total_mentions,
            }
        )

    distribution_path = RESULTS_ROOT / "conll2003_train_entity_distribution_long.csv"
    summary_path = RESULTS_ROOT / "conll2003_train_entity_distribution_summary.csv"

    write_csv(distribution_path, rows)
    write_csv(summary_path, summary_rows)

    print(f"Saved entity distribution: {distribution_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()