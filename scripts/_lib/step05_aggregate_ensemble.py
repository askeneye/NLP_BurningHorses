from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts._lib.repro_io import read_json_lines, write_json


def _relative(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(resolved_root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def resolve_step05_paths(root: Path, config: dict[str, Any]) -> dict[str, Path | str]:
    split_name = str(config.get("default_profile", {}).get("split_name", "k5_seed242"))
    paths = config.get("paths", {})
    interim_root = root / str(paths.get("interim_root", "reproduction/data/interim"))
    logs_root = root / str(paths.get("logs_root", "reproduction/logs"))
    return {
        "split_name": split_name,
        "interim_root": interim_root,
        "logs_root": logs_root,
        "kshot_metadata": interim_root / split_name / "base" / "metadata.json",
        "step04_manifest": logs_root / "step04_predict_bart_manifest.json",
        "step05_manifest": logs_root / "step05_aggregate_ensemble_manifest.json",
        "output_root": interim_root / split_name / "ensemble_labels",
    }


def parse_eval_splits(eval_splits_csv: str | None) -> list[str]:
    if not eval_splits_csv:
        return ["mini_val", "validation", "test", "unlabeled_pool"]
    parsed = [part.strip() for part in eval_splits_csv.split(",") if part.strip()]
    if not parsed:
        raise ValueError("Expected at least one split in --eval-splits.")
    return parsed


def _load_label_list(metadata_path: Path) -> list[str]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    label_list = metadata.get("label_list")
    if not isinstance(label_list, list) or not label_list:
        raise ValueError(f"{metadata_path} must contain a non-empty label_list.")
    return [str(label) for label in label_list]


def load_step05_inputs(
    *,
    step04_manifest_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    if not step04_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing Step-04 run manifest at {step04_manifest_path}. "
            "Run step 04 first (or rerun it) before step 05."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing k-shot metadata file: {metadata_path}")
    return {
        "step04_manifest": json.loads(step04_manifest_path.read_text(encoding="utf-8")),
        "label_list": _load_label_list(metadata_path),
    }


def _normalized_distribution(counts: Counter[str], label_list: list[str], member_count: int) -> list[float]:
    return [counts.get(label, 0) / member_count for label in label_list]


def _temperature_distribution(probabilities: list[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    tempered = [probability ** (1.0 / temperature) if probability > 0 else 0.0 for probability in probabilities]
    total = sum(tempered)
    if total == 0.0:
        return probabilities
    return [value / total for value in tempered]


def _argmax_label(probabilities: list[float], label_list: list[str]) -> tuple[str, float]:
    best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    return label_list[best_index], probabilities[best_index]


def _validate_member_rows(member_rows: list[list[dict[str, Any]]], member_paths: list[Path]) -> None:
    if not member_rows:
        raise ValueError("At least one member file is required for aggregation.")
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


def build_aggregation_plan(
    *,
    resolved_paths: dict[str, Path | str],
    inputs: dict[str, Any],
    eval_splits: list[str],
    pattern_limit: int | None,
) -> dict[str, Any]:
    split_name = str(resolved_paths["split_name"])
    jobs = [job for job in inputs["step04_manifest"].get("jobs", []) if str(job.get("split_name")) == split_name]
    if pattern_limit is not None:
        jobs = jobs[:pattern_limit]
    if not jobs:
        raise ValueError("No Step-04 jobs found for requested split/pattern selection.")

    patterns = [str(job["pattern_id"]) for job in jobs]
    split_member_files: dict[str, list[Path]] = {eval_split: [] for eval_split in eval_splits}
    for job in jobs:
        output_root = Path(job["output_root"])
        for eval_split in eval_splits:
            member_file = output_root / eval_split / "hard_predictions.jsonl"
            split_member_files[eval_split].append(member_file)

    return {"patterns": patterns, "split_member_files": split_member_files}


def _aggregate_one_split(
    *,
    member_paths: list[Path],
    label_list: list[str],
    temperature: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    missing = [path for path in member_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing member prediction files:\n{missing_text}")

    member_rows = [read_json_lines(path) for path in member_paths]
    _validate_member_rows(member_rows, member_paths)

    base_rows = member_rows[0]
    member_count = len(member_rows)
    normalized_rows: list[dict[str, Any]] = []
    temp_rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []

    for row_index, base_row in enumerate(base_rows):
        token_vote_counts: list[dict[str, int]] = []
        normalized_soft_labels: list[list[float]] = []
        temp_soft_labels: list[list[float]] = []
        hard_tags: list[str] = []
        vote_confidences: list[float] = []
        tokens = base_row["tokens"]

        for token_index in range(len(tokens)):
            counts = Counter(member[row_index]["predicted_tags"][token_index] for member in member_rows)
            normalized = _normalized_distribution(counts, label_list, member_count)
            tempered = _temperature_distribution(normalized, temperature)
            hard_label, confidence = _argmax_label(normalized, label_list)
            token_vote_counts.append({label: counts[label] for label in label_list if counts.get(label, 0)})
            normalized_soft_labels.append(normalized)
            temp_soft_labels.append(tempered)
            hard_tags.append(hard_label)
            vote_confidences.append(confidence)

        common: dict[str, Any] = {
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

    return normalized_rows, temp_rows, hard_rows


def run_aggregation(
    *,
    root: Path,
    resolved_paths: dict[str, Path | str],
    plan: dict[str, Any],
    label_list: list[str],
    temperature: float,
    overwrite: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    split_name = str(resolved_paths["split_name"])
    output_root = Path(resolved_paths["output_root"])
    summaries: list[dict[str, Any]] = []

    for eval_split, member_paths in plan["split_member_files"].items():
        split_output_dir = output_root / eval_split
        normalized_path = split_output_dir / "soft_labels_normalized.jsonl"
        temp_path = split_output_dir / f"soft_labels_temp{temperature:g}.jsonl"
        hard_path = split_output_dir / "hard_labels.jsonl"
        manifest_path = split_output_dir / "manifest.json"

        print(
            f"aggregate_split split={split_name} eval={eval_split} "
            f"members={len(member_paths)} output={split_output_dir}"
        )

        if dry_run:
            summaries.append(
                {
                    "split_name": split_name,
                    "eval_split": eval_split,
                    "status": "dry_run",
                    "member_count": len(member_paths),
                    "member_files": [_relative(root, path) for path in member_paths],
                    "outputs": {
                        "vote_normalized": _relative(root, normalized_path),
                        f"vote_temp{temperature:g}": _relative(root, temp_path),
                        "hard_argmax": _relative(root, hard_path),
                    },
                }
            )
            continue

        if not overwrite and normalized_path.exists() and temp_path.exists() and hard_path.exists() and manifest_path.exists():
            summaries.append(
                {
                    "split_name": split_name,
                    "eval_split": eval_split,
                    "status": "skipped_existing",
                    "member_count": len(member_paths),
                    "outputs": {
                        "vote_normalized": _relative(root, normalized_path),
                        f"vote_temp{temperature:g}": _relative(root, temp_path),
                        "hard_argmax": _relative(root, hard_path),
                    },
                }
            )
            continue

        normalized_rows, temp_rows, hard_rows = _aggregate_one_split(
            member_paths=member_paths,
            label_list=label_list,
            temperature=temperature,
        )
        split_output_dir.mkdir(parents=True, exist_ok=True)
        with normalized_path.open("w", encoding="utf-8") as output_file:
            for row in normalized_rows:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        with temp_path.open("w", encoding="utf-8") as output_file:
            for row in temp_rows:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        with hard_path.open("w", encoding="utf-8") as output_file:
            for row in hard_rows:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

        split_manifest = {
            "dataset": "conll2003",
            "kshot_split": split_name,
            "eval_split": eval_split,
            "patterns": list(plan["patterns"]),
            "member_count": len(member_paths),
            "label_list": label_list,
            "temperature": temperature,
            "rows_written": len(hard_rows),
            "source_files": [_relative(root, path) for path in member_paths],
            "outputs": {
                "vote_normalized": _relative(root, normalized_path),
                f"vote_temp{temperature:g}": _relative(root, temp_path),
                "hard_argmax": _relative(root, hard_path),
            },
        }
        write_json(manifest_path, split_manifest)
        summaries.append(
            {
                "split_name": split_name,
                "eval_split": eval_split,
                "status": "written",
                "rows_written": len(hard_rows),
                "member_count": len(member_paths),
                "outputs": split_manifest["outputs"],
            }
        )

    return summaries


def write_step05_run_manifest(
    *,
    manifest_path: Path,
    split_name: str,
    plan: dict[str, Any],
    summaries: list[dict[str, Any]],
    temperature: float,
    dry_run: bool,
) -> Path:
    payload = {
        "split_name": split_name,
        "patterns": list(plan["patterns"]),
        "temperature": temperature,
        "dry_run": dry_run,
        "summaries": summaries,
    }
    write_json(manifest_path, payload)
    return manifest_path


def print_step05_summary(
    summaries: list[dict[str, Any]],
    manifest_path: Path,
    dry_run: bool,
) -> None:
    print("Prepared ensemble teacher aggregation")
    print(f"split_outputs={len(summaries)} dry_run={dry_run}")
    for summary in summaries:
        status = summary.get("status")
        eval_split = summary.get("eval_split")
        member_count = summary.get("member_count")
        rows_written = summary.get("rows_written")
        print(f"eval_split={eval_split} status={status} members={member_count} rows={rows_written}")
    print(f"run_manifest={manifest_path}")
