from __future__ import annotations

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
DEFAULT_INPUT_ROOT = "data/interim/conll2003_ensemble_teachers/bart_base/pet_oada_verbalized_final2000"
DEFAULT_OUTPUT_ROOT = "data/interim/conll2003_ensemble_teachers/bart_base/pet_oada_verbalized_final2000_filtered"
DEFAULT_SPLITS = ["k5_seed42"]
DEFAULT_SUBSET_SIZES = [500, 1000, 2000, 5000]
TEACHER_FILES = {
    "hard_argmax": "ensemble_hard_argmax.jsonl",
    "vote_normalized": "ensemble_vote_normalized.jsonl",
    "vote_temp2": "ensemble_vote_temp2.jsonl",
}


def csv_from_env(name: str, default: list[str]) -> list[str]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def ints_from_env(name: str, default: list[int]) -> list[int]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


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


def validate_parallel_rows(reference_rows: list[dict[str, Any]], other_rows: list[dict[str, Any]], name: str) -> None:
    if len(reference_rows) != len(other_rows):
        raise ValueError(f"{name} row count mismatch: {len(reference_rows)} != {len(other_rows)}")
    for row_index, (reference, other) in enumerate(zip(reference_rows, other_rows)):
        if reference["id"] != other["id"] or reference["tokens"] != other["tokens"]:
            raise ValueError(f"{name} row mismatch at index {row_index}")


def row_agreement_score(row: dict[str, Any]) -> dict[str, Any]:
    predicted_tags = row["predicted_tags"]
    vote_confidences = [float(value) for value in row["vote_confidences"]]
    entity_indices = [index for index, label in enumerate(predicted_tags) if label != "O"]
    has_entity = bool(entity_indices)
    if entity_indices:
        entity_confidence = sum(vote_confidences[index] for index in entity_indices) / len(entity_indices)
        min_entity_confidence = min(vote_confidences[index] for index in entity_indices)
    else:
        entity_confidence = 0.0
        min_entity_confidence = 0.0
    mean_confidence = sum(vote_confidences) / len(vote_confidences) if vote_confidences else 0.0
    unanimous_fraction = (
        sum(1 for confidence in vote_confidences if confidence == 1.0) / len(vote_confidences)
        if vote_confidences
        else 0.0
    )
    return {
        "has_entity": has_entity,
        "entity_token_count": len(entity_indices),
        "entity_confidence": entity_confidence,
        "min_entity_confidence": min_entity_confidence,
        "mean_confidence": mean_confidence,
        "unanimous_fraction": unanimous_fraction,
    }


def selected_indices(hard_rows: list[dict[str, Any]], subset_size: int) -> tuple[list[int], dict[str, Any]]:
    scored_rows = []
    for row_index, row in enumerate(hard_rows):
        score = row_agreement_score(row)
        scored_rows.append((row_index, score))

    entity_rows = [item for item in scored_rows if item[1]["has_entity"]]
    no_entity_rows = [item for item in scored_rows if not item[1]["has_entity"]]
    ranked_entity_rows = sorted(
        entity_rows,
        key=lambda item: (
            item[1]["min_entity_confidence"],
            item[1]["entity_confidence"],
            item[1]["mean_confidence"],
            item[1]["entity_token_count"],
        ),
        reverse=True,
    )
    ranked_no_entity_rows = sorted(
        no_entity_rows,
        key=lambda item: (item[1]["mean_confidence"], item[1]["unanimous_fraction"]),
        reverse=True,
    )

    max_no_entity = max(0, int(subset_size * 0.1))
    selected = ranked_entity_rows[:subset_size]
    if len(selected) < subset_size:
        selected.extend(ranked_no_entity_rows[: subset_size - len(selected)])
    elif max_no_entity:
        selected_entity = selected[: subset_size - max_no_entity]
        selected = selected_entity + ranked_no_entity_rows[:max_no_entity]

    indices = sorted(row_index for row_index, _ in selected[:subset_size])
    selected_scores = [score for _, score in selected[:subset_size]]
    manifest_stats = {
        "requested_subset_size": subset_size,
        "selected_rows": len(indices),
        "selected_with_entities": sum(1 for score in selected_scores if score["has_entity"]),
        "selected_without_entities": sum(1 for score in selected_scores if not score["has_entity"]),
        "mean_entity_confidence": (
            sum(score["entity_confidence"] for score in selected_scores if score["has_entity"])
            / max(1, sum(1 for score in selected_scores if score["has_entity"]))
        ),
        "mean_token_confidence": sum(score["mean_confidence"] for score in selected_scores) / max(1, len(selected_scores)),
        "selection_policy": (
            "Rank entity-containing examples first by min entity-token vote confidence, then mean entity "
            "confidence and mean token confidence. Reserve up to 10% for high-confidence no-entity rows. "
            "Restore original row order before writing."
        ),
    }
    return indices, manifest_stats


def filter_split(split: str, subset_sizes: list[int], input_root: Path, output_root: Path) -> list[dict[str, Any]]:
    started_at = time.time()
    input_dir = input_root / split / "unlabeled_pool"
    manifest = read_json(input_dir / "manifest.json")
    rows_by_teacher = {name: read_jsonl(input_dir / filename) for name, filename in TEACHER_FILES.items()}
    hard_rows = rows_by_teacher["hard_argmax"]
    for name, rows in rows_by_teacher.items():
        validate_parallel_rows(hard_rows, rows, name)

    manifests: list[dict[str, Any]] = []
    for subset_size in subset_sizes:
        indices, subset_stats = selected_indices(hard_rows, subset_size)
        subset_name = f"top_{subset_size}"
        output_dir = output_root / split / "unlabeled_pool" / subset_name
        outputs: dict[str, str] = {}
        for teacher_name, filename in TEACHER_FILES.items():
            selected_rows = [rows_by_teacher[teacher_name][index] for index in indices]
            output_path = output_dir / filename
            write_jsonl(output_path, selected_rows)
            outputs[teacher_name] = repo_relative_path(output_path)
        subset_manifest = {
            "dataset": manifest.get("dataset", "conll2003"),
            "source_method": manifest.get("method"),
            "source_suffix": manifest.get("source_suffix"),
            "kshot_split": split,
            "eval_split": "unlabeled_pool",
            "subset_name": subset_name,
            "source_manifest": repo_relative_path(input_dir / "manifest.json"),
            "source_rows": len(hard_rows),
            "member_count": manifest.get("member_count"),
            "patterns": manifest.get("patterns"),
            "outputs": outputs,
            **subset_stats,
            "elapsed_seconds": time.time() - started_at,
        }
        write_json(output_dir / "manifest.json", subset_manifest)
        manifests.append(subset_manifest)
        print(
            f"{split} {subset_name}: rows={subset_manifest['selected_rows']} "
            f"entities={subset_manifest['selected_with_entities']} "
            f"mean_entity_conf={subset_manifest['mean_entity_confidence']:.3f}"
        )
    return manifests


def main() -> None:
    splits = csv_from_env("FILTER_TEACHER_SPLITS", DEFAULT_SPLITS)
    subset_sizes = ints_from_env("FILTER_TEACHER_SIZES", DEFAULT_SUBSET_SIZES)
    input_root = PROJECT_ROOT / os.environ.get("FILTER_TEACHER_INPUT_ROOT", DEFAULT_INPUT_ROOT)
    output_root = PROJECT_ROOT / os.environ.get("FILTER_TEACHER_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    print("Agreement-filtered teacher subsets")
    print(f"splits={', '.join(splits)}")
    print(f"subset_sizes={', '.join(str(size) for size in subset_sizes)}")
    total = 0
    for split in splits:
        total += len(filter_split(split, subset_sizes, input_root, output_root))
    print(f"Finished filtered teacher subsets: {total} outputs")


if __name__ == "__main__":
    main()
