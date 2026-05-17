from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start_path: Path) -> Path:
    for path in [start_path, *start_path.parents]:
        if (path / "pyproject.toml").exists():
            return path
    raise FileNotFoundError("Could not locate project root containing pyproject.toml.")


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
DEFAULT_SPLIT = "k5_seed42"
DEFAULT_EVAL_SPLIT = "unlabeled_pool"
DEFAULT_PATTERNS = ["pattern_01", "pattern_02", "pattern_03"]
DEFAULT_SUFFIX = "xe_verbalized_pilot1000"
DEFAULT_INPUT_ROOT = "data/interim/conll2003_true_soft_labels/bart_base/pet_oada_verbalized"
DEFAULT_OUTPUT_ROOT = "data/interim/conll2003_ensemble_teachers/bart_base/pet_oada_verbalized_true_soft"
DEFAULT_METADATA_FILE = "data/interim/conll2003_kshot_bert/metadata.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped_line = line.strip()
            if stripped_line:
                rows.append(json.loads(stripped_line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)


def repo_relative_path(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved_path)


def csv_from_env(name: str, default: list[str]) -> list[str]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def label_list_from_metadata(path: Path) -> list[str]:
    labels = read_json(path).get("label_list")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"{path} must contain label_list.")
    return [str(label) for label in labels]


def member_dir(input_root: Path, split: str, pattern: str, suffix: str, eval_split: str) -> Path:
    return input_root / split / f"{pattern}_{suffix}" / eval_split


def validate_rows(member_soft_rows: list[list[dict[str, Any]]], member_hard_rows: list[list[dict[str, Any]]]) -> None:
    if not member_soft_rows or not member_hard_rows:
        raise ValueError("At least one member is required.")
    expected = member_soft_rows[0]
    for rows in [*member_soft_rows[1:], *member_hard_rows]:
        if len(rows) != len(expected):
            raise ValueError("Member row counts do not match.")
        for row_index, (left, right) in enumerate(zip(expected, rows)):
            if left["tokens"] != right["tokens"]:
                raise ValueError(f"Token mismatch at row {row_index}.")


def average_distributions(distributions: list[list[float]]) -> list[float]:
    member_count = len(distributions)
    return [
        sum(distribution[label_index] for distribution in distributions) / member_count
        for label_index in range(len(distributions[0]))
    ]


def argmax_label(distribution: list[float], label_list: list[str]) -> str:
    return label_list[max(range(len(distribution)), key=distribution.__getitem__)]


def aggregate() -> dict[str, Any]:
    started_at = time.time()
    split = os.environ.get("TRUE_SOFT_ENSEMBLE_SPLIT", DEFAULT_SPLIT)
    eval_split = os.environ.get("TRUE_SOFT_ENSEMBLE_EVAL_SPLIT", DEFAULT_EVAL_SPLIT)
    patterns = csv_from_env("TRUE_SOFT_ENSEMBLE_PATTERNS", DEFAULT_PATTERNS)
    suffix = os.environ.get("TRUE_SOFT_ENSEMBLE_SUFFIX", DEFAULT_SUFFIX)
    input_root = PROJECT_ROOT / os.environ.get("TRUE_SOFT_ENSEMBLE_INPUT_ROOT", DEFAULT_INPUT_ROOT)
    output_root = PROJECT_ROOT / os.environ.get("TRUE_SOFT_ENSEMBLE_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    label_list = label_list_from_metadata(
        PROJECT_ROOT / os.environ.get("TRUE_SOFT_ENSEMBLE_METADATA_FILE", DEFAULT_METADATA_FILE)
    )

    member_paths = [member_dir(input_root, split, pattern, suffix, eval_split) for pattern in patterns]
    missing = [
        path
        for member_path in member_paths
        for path in (member_path / "soft_labels.jsonl", member_path / "hard_predictions.jsonl")
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing member files:\n" + "\n".join(str(path) for path in missing))

    member_soft_rows = [read_jsonl(path / "soft_labels.jsonl") for path in member_paths]
    member_hard_rows = [read_jsonl(path / "hard_predictions.jsonl") for path in member_paths]
    validate_rows(member_soft_rows, member_hard_rows)
    member_count = len(member_paths)

    true_soft_rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []
    for row_index, base_row in enumerate(member_soft_rows[0]):
        tokens = base_row["tokens"]
        averaged_soft_labels = [
            average_distributions([member[row_index]["soft_labels"][token_index] for member in member_soft_rows])
            for token_index in range(len(tokens))
        ]
        hard_tags = []
        vote_counts = []
        for token_index in range(len(tokens)):
            counts = Counter(member[row_index]["predicted_tags"][token_index] for member in member_hard_rows)
            vote_counts.append({label: counts[label] for label in label_list if counts.get(label, 0)})
            hard_tags.append(argmax_label([counts.get(label, 0) / member_count for label in label_list], label_list))

        common = {
            "id": base_row.get("id", f"row-{row_index:06d}"),
            "tokens": tokens,
            "member_count": member_count,
        }
        if "ner_tags" in base_row:
            common["ner_tags"] = base_row["ner_tags"]
        if "gold_tags" in base_row:
            common["gold_tags"] = base_row["gold_tags"]
        true_soft_rows.append({**common, "teacher": "verbalized_true_soft", "soft_labels": averaged_soft_labels})
        hard_rows.append(
            {
                **common,
                "teacher": "verbalized_hard_argmax",
                "predicted_tags": hard_tags,
                "vote_counts": vote_counts,
            }
        )

    output_dir = output_root / split / eval_split / f"members_{member_count}"
    true_soft_path = output_dir / "ensemble_verbalized_true_soft.jsonl"
    hard_path = output_dir / "ensemble_verbalized_hard_argmax.jsonl"
    manifest_path = output_dir / "manifest.json"
    write_jsonl(true_soft_path, true_soft_rows)
    write_jsonl(hard_path, hard_rows)
    manifest = {
        "split": split,
        "eval_split": eval_split,
        "patterns": patterns,
        "member_count": member_count,
        "source_suffix": suffix,
        "source_dirs": [repo_relative_path(path) for path in member_paths],
        "outputs": {
            "verbalized_true_soft": repo_relative_path(true_soft_path),
            "verbalized_hard_argmax": repo_relative_path(hard_path),
        },
        "rows_written": len(true_soft_rows),
        "elapsed_seconds": time.time() - started_at,
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    manifest = aggregate()
    print(
        "Finished verbalized true-soft aggregation: "
        f"rows={manifest['rows_written']} members={manifest['member_count']} "
        f"output={manifest['outputs']['verbalized_true_soft']}"
    )


if __name__ == "__main__":
    main()
