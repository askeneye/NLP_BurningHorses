from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from scripts._lib.repro_io import load_yaml, read_json_lines, write_json, write_jsonl


TEXT_COLUMN = "tokens"
LABEL_COLUMN = "ner_tags"


def dataset_rows(hf_split) -> list[dict[str, Any]]:
    return [
        {"id": str(row["id"]), "tokens": row[TEXT_COLUMN], "ner_tags": row[LABEL_COLUMN]}
        for row in hf_split
    ]


def load_train_from_hf() -> tuple[list[dict[str, Any]], list[str]]:
    ds = load_dataset("BramVanroy/conll2003")
    label_list = list(ds["train"].features[LABEL_COLUMN].feature.names)
    return dataset_rows(ds["train"]), label_list


def resolve_step00_paths(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    paths = config.get("paths", {})
    raw_root = root / str(paths.get("raw_data_root", "data/raw/conll2003"))
    data_root = root / str(paths.get("data_root", "reproduction/data"))
    output_root = data_root / "conll2003"
    validation_path = raw_root / "validation.json"
    test_path = raw_root / "test.json"
    return {
        "raw_root": raw_root,
        "output_root": output_root,
        "validation_path": validation_path,
        "test_path": test_path,
    }


def validate_step00_inputs(validation_path: Path, test_path: Path, raw_root: Path) -> None:
    if not validation_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing validation/test json in {raw_root}")


def load_step00_input_rows(validation_path: Path, test_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return read_json_lines(validation_path), read_json_lines(test_path)


def load_legacy_step00_metadata(root: Path) -> tuple[dict[str, Any], Path]:
    legacy_meta_path = root / "data/interim/conll2003_kshot_bert/metadata.json"
    legacy_meta: dict[str, Any] = {}
    if legacy_meta_path.exists():
        legacy_meta = json.loads(legacy_meta_path.read_text(encoding="utf-8"))
    return legacy_meta, legacy_meta_path


def write_step00_outputs(
    output_root: Path,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> None:
    write_jsonl(output_root / "train.jsonl", train_rows)
    write_jsonl(output_root / "validation_raw.jsonl", validation_rows)
    write_jsonl(output_root / "test.jsonl", test_rows)


def build_step00_metadata(
    *,
    root: Path,
    validation_path: Path,
    test_path: Path,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    label_list: list[str],
    legacy_meta: dict[str, Any],
    legacy_meta_path: Path,
) -> dict[str, Any]:
    return {
        "dataset": "conll2003",
        "step": "00_prepare_conll2003",
        "source": {
            "train": "huggingface:BramVanroy/conll2003/train",
            "validation_raw": str(validation_path.relative_to(root)).replace("\\", "/"),
            "test": str(test_path.relative_to(root)).replace("\\", "/"),
        },
        "counts": {"train": len(train_rows), "validation_raw": len(validation_rows), "test": len(test_rows)},
        "label_list": label_list,
        "verification_against_legacy": {
            "legacy_metadata_path": str(legacy_meta_path.relative_to(root)).replace("\\", "/")
            if legacy_meta
            else None,
            "validation_raw_matches_legacy": (
                len(validation_rows) == int(legacy_meta.get("num_raw_validation_sentences", -1))
                if legacy_meta
                else None
            ),
            "test_matches_legacy": (
                len(test_rows) == int(legacy_meta.get("num_test_sentences", -1))
                if legacy_meta
                else None
            ),
            "label_list_matches_legacy": label_list == list(legacy_meta.get("label_list", [])) if legacy_meta else None,
        },
    }


def print_step00_summary(
    output_root: Path,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    legacy_meta: dict[str, Any],
) -> None:
    print("Prepared CoNLL-2003 canonical data")
    print(f"Output root: {output_root}")
    print(f"train={len(train_rows)} validation_raw={len(validation_rows)} test={len(test_rows)}")
    if legacy_meta:
        check = metadata["verification_against_legacy"]
        print(
            "legacy_check:"
            f" validation_match={check['validation_raw_matches_legacy']}"
            f" test_match={check['test_matches_legacy']}"
            f" labels_match={check['label_list_matches_legacy']}"
        )


def run_step00_prepare(root: Path, config_path: Path) -> None:
    config = load_yaml(config_path)
    resolved_paths = resolve_step00_paths(root, config)
    validate_step00_inputs(
        resolved_paths["validation_path"],
        resolved_paths["test_path"],
        resolved_paths["raw_root"],
    )
    validation_rows, test_rows = load_step00_input_rows(
        resolved_paths["validation_path"],
        resolved_paths["test_path"],
    )
    train_rows, label_list = load_train_from_hf()
    legacy_meta, legacy_meta_path = load_legacy_step00_metadata(root)

    write_step00_outputs(
        resolved_paths["output_root"],
        train_rows,
        validation_rows,
        test_rows,
    )
    metadata = build_step00_metadata(
        root=root,
        validation_path=resolved_paths["validation_path"],
        test_path=resolved_paths["test_path"],
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        label_list=label_list,
        legacy_meta=legacy_meta,
        legacy_meta_path=legacy_meta_path,
    )
    output_root = resolved_paths["output_root"]
    write_json(output_root / "metadata.json", metadata)
    print_step00_summary(output_root, train_rows, validation_rows, test_rows, metadata, legacy_meta)
