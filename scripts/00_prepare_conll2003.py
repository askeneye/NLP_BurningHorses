from __future__ import annotations

import argparse
from scripts._lib.repro_io import load_yaml, project_root, write_json
from scripts._lib.step00_prepare import (
    build_step00_metadata,
    load_legacy_step00_metadata,
    load_step00_input_rows,
    load_train_from_hf,
    print_step00_summary,
    resolve_step00_paths,
    validate_step00_inputs,
    write_step00_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare canonical CoNLL-2003 data for reproduction.")
    parser.add_argument(
        "--config",
        default="scripts/config/reproduction_default.yaml",
        help="Path to reproduction YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)

    # 1) Resolve raw input and reproduction output paths.
    resolved_paths = resolve_step00_paths(root, config)

    # 2) Validate required raw files and load raw rows.
    validate_step00_inputs(
        resolved_paths["validation_path"],
        resolved_paths["test_path"],
        resolved_paths["raw_root"],
    )
    validation_rows, test_rows = load_step00_input_rows(
        resolved_paths["validation_path"],
        resolved_paths["test_path"],
    )

    # 3) Load canonical train split and label map from HF parquet mirror.
    train_rows, label_list = load_train_from_hf()

    # 4) Load legacy metadata for deterministic parity checks.
    legacy_meta, legacy_meta_path = load_legacy_step00_metadata(root)

    # 5) Write canonical reproduction files.
    write_step00_outputs(
        resolved_paths["output_root"],
        train_rows,
        validation_rows,
        test_rows,
    )

    # 6) Build and write metadata with legacy verification fields.
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
    write_json(resolved_paths["output_root"] / "metadata.json", metadata)

    # 7) Print compact run summary.
    print_step00_summary(
        resolved_paths["output_root"],
        train_rows,
        validation_rows,
        test_rows,
        metadata,
        legacy_meta,
    )


if __name__ == "__main__":
    main()
