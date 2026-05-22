from __future__ import annotations

import argparse
from scripts._lib.repro_io import load_yaml, project_root
from scripts._lib.step01_kshot import (
    build_shared_eval_splits,
    create_support_and_unlabeled_pool,
    load_step01_inputs,
    print_step01_summary,
    resolve_step01_paths,
    verify_step01_legacy_train,
    write_step01_metadata,
    write_step01_split_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create reproduction k-shot split artifacts.")
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

    # 1) Resolve step inputs/outputs and default split profile.
    resolved = resolve_step01_paths(root, config)

    # 2) Load Step-00 artifacts and label metadata.
    train_rows, validation_raw_rows, test_rows, label_list = load_step01_inputs(resolved["step00_root"])

    # 3) Build shared validation/mini-val/test artifacts.
    validation_core, mini_val_core, test_core = build_shared_eval_splits(validation_raw_rows, test_rows)

    # 4) Build support and unlabeled-pool split with deterministic seed.
    support, rest = create_support_and_unlabeled_pool(
        train_rows,
        int(resolved["k_shot"]),
        int(resolved["split_seed"]),
        label_list,
    )

    # 5) Write split files and metadata.
    write_step01_split_files(
        resolved["output_root"],
        resolved["output_split_dir"],
        validation_core,
        mini_val_core,
        test_core,
        support,
        rest,
    )
    write_step01_metadata(
        resolved["output_root"],
        resolved["output_split_dir"],
        k_shot=int(resolved["k_shot"]),
        split_seed=int(resolved["split_seed"]),
        label_list=label_list,
        validation_core=validation_core,
        mini_val_core=mini_val_core,
        test_core=test_core,
        validation_raw_rows=validation_raw_rows,
        support=support,
        rest=rest,
    )

    # 6) Verify legacy parity and print summary.
    exact_match = verify_step01_legacy_train(root, str(resolved["split_name"]), support)
    print_step01_summary(
        resolved["output_root"],
        str(resolved["split_name"]),
        support,
        rest,
        validation_core,
        mini_val_core,
        test_core,
        exact_match,
    )


if __name__ == "__main__":
    main()
