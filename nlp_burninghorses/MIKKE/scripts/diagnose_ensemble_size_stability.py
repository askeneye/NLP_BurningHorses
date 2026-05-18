from __future__ import annotations

from collections import Counter
import itertools
import json
import os
from pathlib import Path
import random
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start_path: Path) -> Path:
    for path in [start_path, *start_path.parents]:
        if (path / "pyproject.toml").exists():
            return path
    raise FileNotFoundError("Could not locate project root containing pyproject.toml.")


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
DEFAULT_SPLITS = ["k5_seed42", "k10_seed42", "k20_seed42", "k50_seed42"]
DEFAULT_MEMBER_SIZES = [3, 5, 7]
DEFAULT_DRAWS = 5
DEFAULT_SUBSET_SIZE = 1000
DEFAULT_INPUT_ROOT = "data/interim/conll2003_soft_labels/bart_base/pet_oada_verbalized"
DEFAULT_FULL_TEACHER_ROOT = "data/interim/conll2003_ensemble_teachers/bart_base/pet_oada_verbalized_final2000"
DEFAULT_OUTPUT_DIR = "reports/ensemble_size_stability"
DEFAULT_SUFFIX = "xe_verbalized_final2000"
DEFAULT_METADATA_FILE = "data/interim/conll2003_kshot_bert/metadata.json"


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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_list_from_metadata(path: Path) -> list[str]:
    labels = read_json(path).get("label_list")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"{path} must contain label_list.")
    return [str(label) for label in labels]


def validate_member_rows(member_rows: list[list[dict[str, Any]]]) -> None:
    expected = member_rows[0]
    for rows in member_rows[1:]:
        if len(rows) != len(expected):
            raise ValueError("Member row-count mismatch.")
        for row_index, (left, right) in enumerate(zip(expected, rows)):
            if left["tokens"] != right["tokens"]:
                raise ValueError(f"Token mismatch at row {row_index}.")


def member_path(input_root: Path, split: str, pattern: str, suffix: str) -> Path:
    return input_root / split / f"{pattern}_{suffix}" / "unlabeled_pool" / "hard_predictions.jsonl"


def row_score(predicted_tags: list[str], vote_confidences: list[float]) -> dict[str, Any]:
    entity_indices = [index for index, label in enumerate(predicted_tags) if label != "O"]
    has_entity = bool(entity_indices)
    if entity_indices:
        entity_confidence = sum(vote_confidences[index] for index in entity_indices) / len(entity_indices)
        min_entity_confidence = min(vote_confidences[index] for index in entity_indices)
    else:
        entity_confidence = 0.0
        min_entity_confidence = 0.0
    mean_confidence = sum(vote_confidences) / len(vote_confidences) if vote_confidences else 0.0
    return {
        "has_entity": has_entity,
        "entity_token_count": len(entity_indices),
        "entity_confidence": entity_confidence,
        "min_entity_confidence": min_entity_confidence,
        "mean_confidence": mean_confidence,
    }


def aggregate_member_subset(
    member_rows: list[list[dict[str, Any]]],
    member_indices: list[int],
    label_list: list[str],
) -> list[dict[str, Any]]:
    selected_members = [member_rows[index] for index in member_indices]
    member_count = len(selected_members)
    rows: list[dict[str, Any]] = []
    for row_index, base_row in enumerate(selected_members[0]):
        predicted_tags: list[str] = []
        vote_confidences: list[float] = []
        vote_counts: list[dict[str, int]] = []
        for token_index in range(len(base_row["tokens"])):
            counts = Counter(member[row_index]["predicted_tags"][token_index] for member in selected_members)
            probabilities = [counts.get(label, 0) / member_count for label in label_list]
            best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
            predicted_tags.append(label_list[best_index])
            vote_confidences.append(probabilities[best_index])
            vote_counts.append({label: counts[label] for label in label_list if counts.get(label, 0)})
        rows.append(
            {
                "id": base_row.get("id", f"row-{row_index:06d}"),
                "tokens": base_row["tokens"],
                "predicted_tags": predicted_tags,
                "vote_confidences": vote_confidences,
                "vote_counts": vote_counts,
            }
        )
    return rows


def top_indices(rows: list[dict[str, Any]], subset_size: int) -> list[int]:
    scored = [
        (row_index, row_score(row["predicted_tags"], [float(value) for value in row["vote_confidences"]]))
        for row_index, row in enumerate(rows)
    ]
    entity_rows = [item for item in scored if item[1]["has_entity"]]
    no_entity_rows = [item for item in scored if not item[1]["has_entity"]]
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
    ranked_no_entity_rows = sorted(no_entity_rows, key=lambda item: item[1]["mean_confidence"], reverse=True)
    max_no_entity = max(0, int(subset_size * 0.1))
    selected = ranked_entity_rows[:subset_size]
    if len(selected) < subset_size:
        selected.extend(ranked_no_entity_rows[: subset_size - len(selected)])
    elif max_no_entity:
        selected = selected[: subset_size - max_no_entity] + ranked_no_entity_rows[:max_no_entity]
    return sorted(row_index for row_index, _ in selected[:subset_size])


def spans(tags: list[str]) -> set[tuple[int, int, str]]:
    output: set[tuple[int, int, str]] = set()
    start: int | None = None
    entity_type: str | None = None
    for index, tag in enumerate([*tags, "O"]):
        if tag == "O" or "-" not in tag:
            if start is not None and entity_type is not None:
                output.add((start, index, entity_type))
            start = None
            entity_type = None
            continue
        prefix, current_type = tag.split("-", 1)
        if prefix == "B" or entity_type != current_type:
            if start is not None and entity_type is not None:
                output.add((start, index, entity_type))
            start = index
            entity_type = current_type
    return output


def compare_subsets(
    candidate_rows: list[dict[str, Any]],
    candidate_indices: list[int],
    full_rows: list[dict[str, Any]],
    full_indices: list[int],
) -> dict[str, float]:
    candidate_set = set(candidate_indices)
    full_set = set(full_indices)
    shared = sorted(candidate_set.intersection(full_set))
    union_size = len(candidate_set.union(full_set))

    token_matches = 0
    token_total = 0
    span_f1_values: list[float] = []
    for row_index in shared:
        candidate_tags = candidate_rows[row_index]["predicted_tags"]
        full_tags = full_rows[row_index]["predicted_tags"]
        token_total += len(full_tags)
        token_matches += sum(1 for left, right in zip(candidate_tags, full_tags) if left == right)
        candidate_spans = spans(candidate_tags)
        full_spans = spans(full_tags)
        tp = len(candidate_spans.intersection(full_spans))
        precision = tp / len(candidate_spans) if candidate_spans else (1.0 if not full_spans else 0.0)
        recall = tp / len(full_spans) if full_spans else (1.0 if not candidate_spans else 0.0)
        span_f1_values.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))

    candidate_scores = [
        row_score(candidate_rows[index]["predicted_tags"], candidate_rows[index]["vote_confidences"])
        for index in candidate_indices
    ]
    return {
        "row_overlap": len(shared),
        "row_jaccard": len(shared) / union_size if union_size else 0.0,
        "token_agreement_on_overlap": token_matches / token_total if token_total else 0.0,
        "mean_span_f1_on_overlap": sum(span_f1_values) / len(span_f1_values) if span_f1_values else 0.0,
        "selected_with_entities": sum(1 for score in candidate_scores if score["has_entity"]),
        "mean_entity_confidence": (
            sum(score["entity_confidence"] for score in candidate_scores if score["has_entity"])
            / max(1, sum(1 for score in candidate_scores if score["has_entity"]))
        ),
        "mean_token_confidence": sum(score["mean_confidence"] for score in candidate_scores) / len(candidate_scores),
    }


def member_draws(member_count: int, member_size: int, draws: int, rng: random.Random) -> list[list[int]]:
    if member_size > member_count:
        return []
    deterministic = list(range(member_size))
    output = [deterministic]
    seen = {tuple(deterministic)}
    all_combinations = list(itertools.combinations(range(member_count), member_size))
    while len(output) < min(draws + 1, len(all_combinations)):
        draw = sorted(rng.sample(range(member_count), member_size))
        key = tuple(draw)
        if key in seen:
            continue
        seen.add(key)
        output.append(draw)
    return output


def diagnose_split(
    split: str,
    member_sizes: list[int],
    draws: int,
    subset_size: int,
    input_root: Path,
    full_teacher_root: Path,
    suffix: str,
    label_list: list[str],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(full_teacher_root / split / "unlabeled_pool" / "manifest.json")
    patterns = [str(pattern) for pattern in manifest["patterns"]]
    member_rows = [read_jsonl(member_path(input_root, split, pattern, suffix)) for pattern in patterns]
    validate_member_rows(member_rows)
    full_rows = read_jsonl(full_teacher_root / split / "unlabeled_pool" / "ensemble_hard_argmax.jsonl")
    full_indices = top_indices(full_rows, subset_size)

    rows: list[dict[str, Any]] = []
    for member_size in member_sizes:
        for draw_index, members in enumerate(member_draws(len(patterns), member_size, draws, rng)):
            candidate_rows = aggregate_member_subset(member_rows, members, label_list)
            candidate_indices = top_indices(candidate_rows, subset_size)
            metrics = compare_subsets(candidate_rows, candidate_indices, full_rows, full_indices)
            rows.append(
                {
                    "split": split,
                    "member_size": member_size,
                    "draw_index": draw_index,
                    "members": [patterns[index] for index in members],
                    **metrics,
                }
            )

    summary: dict[str, Any] = {
        "split": split,
        "full_member_count": len(patterns),
        "subset_size": subset_size,
        "candidate_draws": len(rows),
        "by_member_size": {},
    }
    for member_size in member_sizes:
        matching = [row for row in rows if row["member_size"] == member_size]
        if not matching:
            continue
        summary["by_member_size"][str(member_size)] = {
            "draws": len(matching),
            "mean_row_overlap": sum(row["row_overlap"] for row in matching) / len(matching),
            "mean_row_jaccard": sum(row["row_jaccard"] for row in matching) / len(matching),
            "mean_token_agreement_on_overlap": sum(row["token_agreement_on_overlap"] for row in matching) / len(matching),
            "mean_span_f1_on_overlap": sum(row["mean_span_f1_on_overlap"] for row in matching) / len(matching),
            "mean_entity_confidence": sum(row["mean_entity_confidence"] for row in matching) / len(matching),
        }
    return rows, summary


def main() -> None:
    splits = csv_from_env("ENSEMBLE_SIZE_SPLITS", DEFAULT_SPLITS)
    member_sizes = ints_from_env("ENSEMBLE_SIZE_MEMBER_SIZES", DEFAULT_MEMBER_SIZES)
    draws = int(os.environ.get("ENSEMBLE_SIZE_DRAWS", DEFAULT_DRAWS))
    subset_size = int(os.environ.get("ENSEMBLE_SIZE_SUBSET_SIZE", DEFAULT_SUBSET_SIZE))
    seed = int(os.environ.get("ENSEMBLE_SIZE_SEED", "42"))
    input_root = PROJECT_ROOT / os.environ.get("ENSEMBLE_SIZE_INPUT_ROOT", DEFAULT_INPUT_ROOT)
    full_teacher_root = PROJECT_ROOT / os.environ.get("ENSEMBLE_SIZE_FULL_TEACHER_ROOT", DEFAULT_FULL_TEACHER_ROOT)
    output_dir = PROJECT_ROOT / os.environ.get("ENSEMBLE_SIZE_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    suffix = os.environ.get("ENSEMBLE_SIZE_SUFFIX", DEFAULT_SUFFIX)
    label_list = label_list_from_metadata(PROJECT_ROOT / os.environ.get("ENSEMBLE_SIZE_METADATA_FILE", DEFAULT_METADATA_FILE))
    rng = random.Random(seed)

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for split in splits:
        rows, summary = diagnose_split(
            split,
            member_sizes,
            draws,
            subset_size,
            input_root,
            full_teacher_root,
            suffix,
            label_list,
            rng,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print(f"{split}:")
        for member_size, stats in summary["by_member_size"].items():
            print(
                f"  members={member_size} overlap={stats['mean_row_overlap']:.1f}/{subset_size} "
                f"jaccard={stats['mean_row_jaccard']:.3f} "
                f"token_agree={stats['mean_token_agreement_on_overlap']:.3f} "
                f"span_f1={stats['mean_span_f1_on_overlap']:.3f}"
            )

    write_jsonl(output_dir / "draw_metrics.jsonl", all_rows)
    write_json(
        output_dir / "summary.json",
        {
            "splits": splits,
            "member_sizes": member_sizes,
            "draws_per_size_excluding_deterministic": draws,
            "subset_size": subset_size,
            "seed": seed,
            "summaries": summaries,
        },
    )
    print(f"Finished ensemble-size diagnostics: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
