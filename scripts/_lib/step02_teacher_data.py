from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts._lib.repro_io import load_yaml, read_json_lines, write_json, write_jsonl
from scripts._lib.teacher_data import (
    apply_pattern,
    canonical_pattern_id,
    detokenize_tokens,
    entity_schema_from_label_list,
    extract_entities_by_type,
    extract_entities_left_to_right,
    generate_target_by_order,
    generate_target_left_to_right,
    normalize_pattern_bank,
    oada_permutations,
    verbalize_entity_order,
)


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def resolve_step02_paths(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    default_profile = config.get("default_profile", {})
    split_name = str(default_profile.get("split_name", "k5_seed242"))
    paths = config.get("paths", {})
    interim_root = root / str(paths.get("interim_root", "reproduction/data/interim"))
    pattern_file = root / str(paths.get("pattern_bank_file", "scripts/assets/patterns.yaml"))
    split_root = interim_root / split_name
    kshot_root = split_root / "base"
    teacher_data_root = split_root / "teacher_data"
    return {
        "split_name": split_name,
        "pattern_file": pattern_file,
        "split_root": split_root,
        "kshot_root": kshot_root,
        "teacher_data_root": teacher_data_root,
        "train_output_dir": teacher_data_root / "train",
        "inference_output_dir": teacher_data_root / "inference",
    }


def load_step02_inputs(kshot_root: Path, split_name: str) -> dict[str, Any]:
    train_rows = read_json_lines(kshot_root / "train.jsonl")
    unlabeled_pool_rows = read_json_lines(kshot_root / "unlabeled_pool.jsonl")
    mini_val_rows = read_json_lines(kshot_root / "mini_val.jsonl")
    validation_rows = read_json_lines(kshot_root / "validation.jsonl")
    test_rows = read_json_lines(kshot_root / "test.jsonl")
    kshot_meta = json.loads((kshot_root / "metadata.json").read_text(encoding="utf-8"))
    return {
        "train_rows": train_rows,
        "unlabeled_pool_rows": unlabeled_pool_rows,
        "mini_val_rows": mini_val_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "label_list": list(kshot_meta["label_list"]),
    }


def select_teacher_patterns(
    pattern_file: Path,
    ensemble_size: int,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    all_patterns = normalize_pattern_bank(load_yaml(pattern_file), "patterns")
    selected_patterns = all_patterns[:ensemble_size]
    if not selected_patterns:
        raise ValueError("Pattern bank is empty; cannot build teacher data.")
    return selected_patterns, selected_patterns[0]


def build_teacher_train_files(
    root: Path,
    train_rows: list[dict[str, Any]],
    split_name: str,
    patterns: list[dict[str, str]],
    entity_schema: list[str],
    id_to_label: dict[int, str],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    permutations = oada_permutations(entity_schema)
    train_manifest: dict[str, dict[str, Any]] = {}

    for pattern_index, pattern in enumerate(patterns, start=1):
        pattern_id = canonical_pattern_id(pattern_index)
        pattern_path = output_dir / f"{pattern_id}.jsonl"
        rows: list[dict[str, str]] = []
        for row in train_rows:
            sentence_text = detokenize_tokens(row["tokens"])
            entities_by_type = extract_entities_by_type(row["tokens"], row["ner_tags"], id_to_label)
            for order in permutations:
                rows.append(
                    {
                        "input_text": apply_pattern(sentence_text, verbalize_entity_order(order), pattern["template"]),
                        "target_text": generate_target_by_order(entities_by_type, order),
                    }
                )
        write_jsonl(pattern_path, rows)
        train_manifest[pattern_id] = {
            "source_pattern_id": pattern["id"],
            "source_pattern_type": pattern.get("type"),
            "source_template": pattern["template"],
            "source_train_file": f"reproduction/data/interim/{split_name}/base/train.jsonl",
            "output_file": _relative(root, pattern_path),
            "method": "pet_oada",
            "source_rows": len(train_rows),
            "generated_rows": len(rows),
        }
    write_json(output_dir / "manifest.json", train_manifest)
    return train_manifest


def build_teacher_inference_files(
    root: Path,
    inference_splits: dict[str, list[dict[str, Any]]],
    pattern: dict[str, str],
    id_to_label: dict[int, str],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "inference_order": "first to last",
        "source_pattern_id": pattern["id"],
        "source_pattern_type": pattern.get("type"),
        "source_template": pattern["template"],
        "splits": {},
    }
    for split_name, rows in inference_splits.items():
        output_file = output_dir / f"{split_name}.jsonl"
        generated = []
        for row in rows:
            sentence_text = detokenize_tokens(row["tokens"])
            entities = extract_entities_left_to_right(row["tokens"], row["ner_tags"], id_to_label)
            generated.append(
                {
                    "input_text": apply_pattern(sentence_text, "first to last", pattern["template"]),
                    "target_text": generate_target_left_to_right(entities),
                }
            )
        write_jsonl(output_file, generated)
        manifest["splits"][split_name] = {
            "source_rows": len(rows),
            "generated_rows": len(generated),
            "output_file": _relative(root, output_file),
        }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def verify_pattern01_against_legacy(root: Path, split_name: str, train_output_dir: Path) -> bool | None:
    legacy_pattern_01 = (
        root / "data/interim/conll2003_kshot_seq2seq/train/pet_oada_verbalized" / split_name / "pattern_01.jsonl"
    )
    if not legacy_pattern_01.exists():
        return None
    return read_json_lines(train_output_dir / "pattern_01.jsonl") == read_json_lines(legacy_pattern_01)


def print_step02_summary(
    split_name: str,
    selected_patterns: list[dict[str, str]],
    train_manifest: dict[str, dict[str, Any]],
    inference_manifest: dict[str, Any],
    legacy_match: bool | None,
) -> None:
    print("Built BART teacher data")
    print(f"split={split_name} patterns={len(selected_patterns)}")
    print(
        "train_rows_per_pattern="
        + ", ".join(f"{pattern_id}:{record['generated_rows']}" for pattern_id, record in train_manifest.items())
    )
    print(
        "inference_rows="
        + ", ".join(f"{name}:{record['generated_rows']}" for name, record in inference_manifest["splits"].items())
    )
    if legacy_match is not None:
        print(f"legacy_pattern_01_exact_match={legacy_match}")
