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

DEFAULT_KSHOT_SPLITS = ["k5_seed42", "k5_seed142", "k5_seed242"]
DEFAULT_PATTERNS = [f"pattern_{index:02d}" for index in range(1, 11)]
DEFAULT_EVAL_SPLITS = ["mini_val", "validation"]
DEFAULT_SUFFIX = "xe_improved_patterns_fixed2000"
DEFAULT_INPUT_ROOT = "data/interim/conll2003_soft_labels/bart_base/pet_oada"
DEFAULT_OUTPUT_ROOT = "data/interim/conll2003_ensemble_teachers/bart_base/pet_oada"
DEFAULT_METADATA_FILE = "data/interim/conll2003_kshot_bert/metadata.json"
DEFAULT_TEMPERATURE = 2.0


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


def label_list_from_metadata(metadata_path: Path) -> list[str]:
    metadata = read_json(metadata_path)
    label_list = metadata.get("label_list")
    if not isinstance(label_list, list) or not label_list:
        raise ValueError(f"{metadata_path} must contain a non-empty label_list.")
    return [str(label) for label in label_list]


def hard_prediction_file(input_root: Path, split: str, pattern: str, suffix: str, eval_split: str) -> Path:
    return input_root / split / f"{pattern}_{suffix}" / eval_split / "hard_predictions.jsonl"


def validate_member_rows(member_rows: list[list[dict[str, Any]]], member_paths: list[Path]) -> None:
    if not member_rows:
        raise ValueError("At least one member file is required.")

    expected_rows = member_rows[0]
    for member_index, rows in enumerate(member_rows[1:], start=1):
        if len(rows) != len(expected_rows):
            raise ValueError(
                f"Row count mismatch: {member_paths[0]} has {len(expected_rows)} rows, "
                f"but {member_paths[member_index]} has {len(rows)} rows."
            )

        for row_index, (expected, actual) in enumerate(zip(expected_rows, rows)):
            if expected["tokens"] != actual["tokens"]:
                raise ValueError(
                    f"Token mismatch at row {row_index}: {member_paths[0]} vs {member_paths[member_index]}"
                )


def normalized_distribution(counts: Counter[str], label_list: list[str], member_count: int) -> list[float]:
    return [counts.get(label, 0) / member_count for label in label_list]


def temperature_distribution(probabilities: list[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    tempered = [probability ** (1.0 / temperature) if probability > 0 else 0.0 for probability in probabilities]
    total = sum(tempered)
    if total == 0:
        return probabilities
    return [value / total for value in tempered]


def argmax_label(probabilities: list[float], label_list: list[str]) -> tuple[str, float]:
    best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    return label_list[best_index], probabilities[best_index]


def aggregate_split(
    split: str,
    eval_split: str,
    patterns: list[str],
    suffix: str,
    input_root: Path,
    output_root: Path,
    label_list: list[str],
    temperature: float,
) -> dict[str, Any]:
    started_at = time.time()
    member_paths = [
        hard_prediction_file(input_root, split, pattern, suffix, eval_split)
        for pattern in patterns
    ]
    missing_paths = [path for path in member_paths if not path.exists()]
    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing hard prediction files:\n{missing_text}")

    member_rows = [read_jsonl(path) for path in member_paths]
    validate_member_rows(member_rows, member_paths)
    member_count = len(member_rows)
    base_rows = member_rows[0]

    normalized_rows: list[dict[str, Any]] = []
    temp_rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []

    for row_index, base_row in enumerate(base_rows):
        tokens = base_row["tokens"]
        token_vote_counts: list[dict[str, int]] = []
        normalized_soft_labels: list[list[float]] = []
        temp_soft_labels: list[list[float]] = []
        hard_tags: list[str] = []
        vote_confidences: list[float] = []

        for token_index in range(len(tokens)):
            counts = Counter(member[row_index]["predicted_tags"][token_index] for member in member_rows)
            normalized = normalized_distribution(counts, label_list, member_count)
            tempered = temperature_distribution(normalized, temperature)
            hard_label, confidence = argmax_label(normalized, label_list)

            token_vote_counts.append({label: counts[label] for label in label_list if counts.get(label, 0)})
            normalized_soft_labels.append(normalized)
            temp_soft_labels.append(tempered)
            hard_tags.append(hard_label)
            vote_confidences.append(confidence)

        common = {
            "id": base_row.get("id", f"row-{row_index:06d}"),
            "tokens": tokens,
            "member_count": member_count,
        }
        if "ner_tags" in base_row:
            common["ner_tags"] = base_row["ner_tags"]
        if "gold_tags" in base_row:
            common["gold_tags"] = base_row["gold_tags"]

        normalized_rows.append(
            {
                **common,
                "teacher": "vote_normalized",
                "soft_labels": normalized_soft_labels,
                "vote_counts": token_vote_counts,
            }
        )
        temp_rows.append(
            {
                **common,
                "teacher": f"vote_temp{temperature:g}",
                "soft_labels": temp_soft_labels,
                "vote_counts": token_vote_counts,
            }
        )
        hard_rows.append(
            {
                **common,
                "teacher": "hard_argmax",
                "predicted_tags": hard_tags,
                "vote_confidences": vote_confidences,
                "vote_counts": token_vote_counts,
            }
        )

    split_output_dir = output_root / split / eval_split
    normalized_path = split_output_dir / "ensemble_vote_normalized.jsonl"
    temp_path = split_output_dir / f"ensemble_vote_temp{temperature:g}.jsonl"
    hard_path = split_output_dir / "ensemble_hard_argmax.jsonl"
    manifest_path = split_output_dir / "manifest.json"

    write_jsonl(normalized_path, normalized_rows)
    write_jsonl(temp_path, temp_rows)
    write_jsonl(hard_path, hard_rows)

    elapsed_seconds = time.time() - started_at
    manifest = {
        "dataset": "conll2003",
        "model_family": "bart_base",
        "method": "pet_oada",
        "kshot_split": split,
        "eval_split": eval_split,
        "patterns": patterns,
        "member_count": member_count,
        "source_suffix": suffix,
        "source_files": [repo_relative_path(path) for path in member_paths],
        "label_list": label_list,
        "temperature": temperature,
        "rows_written": len(base_rows),
        "outputs": {
            "vote_normalized": repo_relative_path(normalized_path),
            f"vote_temp{temperature:g}": repo_relative_path(temp_path),
            "hard_argmax": repo_relative_path(hard_path),
        },
        "elapsed_seconds": elapsed_seconds,
    }
    write_json(manifest_path, manifest)

    return manifest


def main() -> None:
    splits = csv_from_env("ENSEMBLE_TEACHER_SPLITS", DEFAULT_KSHOT_SPLITS)
    patterns = csv_from_env("ENSEMBLE_TEACHER_PATTERNS", DEFAULT_PATTERNS)
    eval_splits = csv_from_env("ENSEMBLE_TEACHER_EVAL_SPLITS", DEFAULT_EVAL_SPLITS)
    suffix = os.environ.get("ENSEMBLE_TEACHER_SOURCE_SUFFIX", DEFAULT_SUFFIX)
    input_root = PROJECT_ROOT / os.environ.get("ENSEMBLE_TEACHER_INPUT_ROOT", DEFAULT_INPUT_ROOT)
    output_root = PROJECT_ROOT / os.environ.get("ENSEMBLE_TEACHER_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    metadata_path = PROJECT_ROOT / os.environ.get("ENSEMBLE_TEACHER_METADATA_FILE", DEFAULT_METADATA_FILE)
    temperature = float(os.environ.get("ENSEMBLE_TEACHER_TEMPERATURE", DEFAULT_TEMPERATURE))
    label_list = label_list_from_metadata(metadata_path)

    print("BART ensemble vote teacher aggregation")
    print(f"splits={', '.join(splits)}")
    print(f"patterns={', '.join(patterns)}")
    print(f"eval_splits={', '.join(eval_splits)}")
    print(f"suffix={suffix}")
    print(f"temperature={temperature:g}")

    total_outputs = 0
    for split in splits:
        for eval_split in eval_splits:
            manifest = aggregate_split(
                split=split,
                eval_split=eval_split,
                patterns=patterns,
                suffix=suffix,
                input_root=input_root,
                output_root=output_root,
                label_list=label_list,
                temperature=temperature,
            )
            total_outputs += 1
            print(
                f"{split} {eval_split}: rows={manifest['rows_written']} "
                f"members={manifest['member_count']} "
                f"output={manifest['outputs']['vote_normalized']}"
            )

    print(f"Finished ensemble teacher aggregation: {total_outputs} split outputs.")


if __name__ == "__main__":
    main()
