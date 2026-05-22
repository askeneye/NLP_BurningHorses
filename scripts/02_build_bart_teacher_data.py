from __future__ import annotations

import argparse
from scripts._lib.repro_io import load_yaml, project_root
from scripts._lib.step02_teacher_data import (
    build_teacher_inference_files,
    build_teacher_train_files,
    load_step02_inputs,
    print_step02_summary,
    resolve_step02_paths,
    select_teacher_patterns,
    verify_pattern01_against_legacy,
)
from scripts._lib.teacher_data import entity_schema_from_label_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BART teacher data for reproduction pipeline.")
    parser.add_argument(
        "--config",
        default="scripts/config/reproduction_default.yaml",
        help="Path to reproduction YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config_path = root / args.config
    config = load_yaml(config_path)

    # 1) Resolve all path locations from config.
    resolved_paths = resolve_step02_paths(root, config)
    split_name = str(resolved_paths["split_name"])

    # 2) Load pre-built k-shot artifacts from Step 01.
    step02_inputs = load_step02_inputs(resolved_paths["kshot_root"], split_name)
    label_list = step02_inputs["label_list"]
    id_to_label = {index: label for index, label in enumerate(label_list)}
    entity_schema = entity_schema_from_label_list(label_list)

    # 3) Select teacher patterns
    ensemble_size = int(config.get("teacher_ensemble", {}).get("ensemble_size", 10))
    selected_patterns, inference_pattern = select_teacher_patterns(
        resolved_paths["pattern_file"],
        ensemble_size=ensemble_size,
    )

    # 4) Build PET+OADA training files for each ensemble pattern.
    train_manifest = build_teacher_train_files(
        root=root,
        train_rows=step02_inputs["train_rows"],
        split_name=split_name,
        patterns=selected_patterns,
        entity_schema=entity_schema,
        id_to_label=id_to_label,
        output_dir=resolved_paths["train_output_dir"],
    )

    # 5) Build first-to-last inference files for scoring/inference splits.
    inference_manifest = build_teacher_inference_files(
        root=root,
        inference_splits={
            "mini_val": step02_inputs["mini_val_rows"],
            "validation": step02_inputs["validation_rows"],
            "test": step02_inputs["test_rows"],
            "unlabeled_pool": step02_inputs["unlabeled_pool_rows"],
        },
        pattern=inference_pattern,
        id_to_label=id_to_label,
        output_dir=resolved_paths["inference_output_dir"],
    )

    # 6) Verify backward parity on pattern_01 where legacy data exists.
    legacy_match = verify_pattern01_against_legacy(
        root=root,
        split_name=split_name,
        train_output_dir=resolved_paths["train_output_dir"],
    )

    # 7) Print compact summary for humans and logs.
    print_step02_summary(
        split_name=split_name,
        selected_patterns=selected_patterns,
        train_manifest=train_manifest,
        inference_manifest=inference_manifest,
        legacy_match=legacy_match,
    )


if __name__ == "__main__":
    main()
