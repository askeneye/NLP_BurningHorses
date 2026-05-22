from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
MIKKE_SCRIPT_DIR = PROJECT_ROOT / "nlp_burninghorses" / "MIKKE" / "scripts"

SPLIT = "k5_seed242"
SUBSET_SIZE = 1000
INPUT_ROOT = PROJECT_ROOT / "data/interim/conll2003_ensemble_teachers/bart_base/pet_oada_verbalized_final1000"
OUTPUT_ROOT = PROJECT_ROOT / "data/interim/conll2003_ensemble_teachers/bart_base/pet_oada_verbalized_final1000_entitymean_filtered"
BERT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "models/bert_conll_distilled_fixed_final600_report"
    / SPLIT
    / "pat_perm_bart_ensemble_to_bert_final1000_entitymean"
    / "hard_argmax"
)

TEACHER_FILES = {
    "hard_argmax": "ensemble_hard_argmax.jsonl",
    "vote_normalized": "ensemble_vote_normalized.jsonl",
    "vote_temp2": "ensemble_vote_temp2.jsonl",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def repo_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


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


def selected_indices_entity_mean_first(hard_rows: list[dict[str, Any]], subset_size: int) -> tuple[list[int], dict[str, Any]]:
    scored_rows = [(row_index, row_agreement_score(row)) for row_index, row in enumerate(hard_rows)]
    entity_rows = [item for item in scored_rows if item[1]["has_entity"]]
    no_entity_rows = [item for item in scored_rows if not item[1]["has_entity"]]

    ranked_entity_rows = sorted(
        entity_rows,
        key=lambda item: (
            item[1]["entity_confidence"],
            item[1]["min_entity_confidence"],
            item[1]["entity_token_count"],
            item[1]["mean_confidence"],
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
        selected = selected[: subset_size - max_no_entity] + ranked_no_entity_rows[:max_no_entity]

    selected = selected[:subset_size]
    indices = sorted(row_index for row_index, _ in selected)
    selected_scores = [score for _, score in selected]
    entity_scores = [score for score in selected_scores if score["has_entity"]]
    stats = {
        "requested_subset_size": subset_size,
        "selected_rows": len(indices),
        "selected_with_entities": len(entity_scores),
        "selected_without_entities": sum(1 for score in selected_scores if not score["has_entity"]),
        "mean_entity_confidence": sum(score["entity_confidence"] for score in entity_scores) / max(1, len(entity_scores)),
        "mean_min_entity_confidence": sum(score["min_entity_confidence"] for score in entity_scores) / max(1, len(entity_scores)),
        "mean_token_confidence": sum(score["mean_confidence"] for score in selected_scores) / max(1, len(selected_scores)),
        "selection_policy": (
            "Diagnostic entity-mean-first filter. Rank entity-containing examples by mean entity-token vote "
            "confidence first, then min entity-token confidence, entity-token count, and all-token mean "
            "confidence. Reserve up to 10% for high-confidence no-entity rows. Restore original row order."
        ),
    }
    return indices, stats


def create_filtered_subset() -> Path:
    started_at = time.time()
    input_dir = INPUT_ROOT / SPLIT / "unlabeled_pool"
    output_dir = OUTPUT_ROOT / SPLIT / "unlabeled_pool" / f"top_{SUBSET_SIZE}"
    manifest = read_json(input_dir / "manifest.json")
    rows_by_teacher = {name: read_jsonl(input_dir / filename) for name, filename in TEACHER_FILES.items()}
    hard_rows = rows_by_teacher["hard_argmax"]
    for name, rows in rows_by_teacher.items():
        validate_parallel_rows(hard_rows, rows, name)

    indices, subset_stats = selected_indices_entity_mean_first(hard_rows, SUBSET_SIZE)
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
        "kshot_split": SPLIT,
        "eval_split": "unlabeled_pool",
        "subset_name": f"top_{SUBSET_SIZE}",
        "source_manifest": repo_relative_path(input_dir / "manifest.json"),
        "source_rows": len(hard_rows),
        "member_count": manifest.get("member_count"),
        "patterns": manifest.get("patterns"),
        "outputs": outputs,
        **subset_stats,
        "elapsed_seconds": time.time() - started_at,
    }
    write_json(output_dir / "manifest.json", subset_manifest)
    print(
        f"{SPLIT} entitymean top_{SUBSET_SIZE}: rows={subset_manifest['selected_rows']} "
        f"entities={subset_manifest['selected_with_entities']} "
        f"mean_entity_conf={subset_manifest['mean_entity_confidence']:.3f} "
        f"mean_min_entity_conf={subset_manifest['mean_min_entity_confidence']:.3f}",
        flush=True,
    )
    return output_dir / "ensemble_hard_argmax.jsonl"


def run_python(script: Path, env: dict[str, str]) -> None:
    merged_env = os.environ.copy()
    merged_env.update(env)
    subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT, env=merged_env, check=True)


def distill(train_file: Path) -> None:
    if (BERT_OUTPUT_DIR / "training_summary.json").exists():
        print(f"Skipping existing BERT distillation: {BERT_OUTPUT_DIR / 'training_summary.json'}", flush=True)
        return
    print("Distilling entity-mean-first final1000 labels to BERT", flush=True)
    run_python(
        PROJECT_ROOT / "nlp_burninghorses" / "bert_conll_distill.py",
        {
            "DISTILL_SPLIT": SPLIT,
            "DISTILL_TEACHER_SPLIT": "unlabeled_pool_top_1000_final1000_entitymean",
            "DISTILL_TEACHER_VARIANT": "hard_argmax",
            "DISTILL_TRAIN_FILE": str(train_file),
            "DISTILL_OUTPUT_DIR": str(BERT_OUTPUT_DIR),
            "DISTILL_MAX_STEPS": "600",
            "DISTILL_EVAL_EVERY": "100",
            "DISTILL_PATIENCE": "5",
            "DISTILL_EARLY_STOPPING": "false",
            "DISTILL_USE_BEST_CHECKPOINT": "false",
            "DISTILL_SEED": "42",
        },
    )


def main() -> None:
    print("Running final1000 entity-mean-first filter/distillation diagnostic", flush=True)
    train_file = create_filtered_subset()
    distill(train_file)
    print(f"Done. Summary: {BERT_OUTPUT_DIR / 'training_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
