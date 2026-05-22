from __future__ import annotations

import argparse

from scripts._lib.repro_io import load_yaml, project_root
from scripts._lib.step04_predict_bart import (
    build_prediction_jobs,
    load_step04_inputs,
    parse_eval_splits,
    print_step04_summary,
    resolve_step04_paths,
    run_prediction_jobs,
    write_step04_run_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hard predictions with trained BART teachers.")
    parser.add_argument(
        "--config",
        default="scripts/config/reproduction_default.yaml",
        help="Path to reproduction YAML config.",
    )
    parser.add_argument(
        "--pattern-limit",
        type=int,
        default=None,
        help="Optionally limit number of teacher patterns to run.",
    )
    parser.add_argument(
        "--eval-splits",
        default="unlabeled_pool",
        help="Comma-separated split names from {unlabeled_pool,mini_val,test,validation}.",
    )
    parser.add_argument(
        "--checkpoint-subdir",
        default=None,
        help="Optional checkpoint subdir override (for example last_step or best_span_f1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Generation batch size per teacher.",
    )
    parser.add_argument(
        "--max-source-length",
        type=int,
        default=128,
        help="Tokenizer max source length.",
    )
    parser.add_argument(
        "--max-generation-length",
        type=int,
        default=128,
        help="Generation max sequence length.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing predictions if present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prediction plan without running inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)

    # 1) Resolve paths and parse requested evaluation splits.
    resolved = resolve_step04_paths(root, config)
    eval_splits = parse_eval_splits(args.eval_splits)

    # 2) Load Step-03 manifest, pattern bank, and split input rows.
    inputs = load_step04_inputs(
        step03_manifest_path=resolved["step03_manifest"],
        kshot_root=resolved["kshot_root"],
        split_name=str(resolved["split_name"]),
        pattern_file=resolved["pattern_file"],
        eval_splits=eval_splits,
    )

    # 3) Build concrete prediction jobs.
    jobs = build_prediction_jobs(
        root=root,
        config=config,
        resolved_paths=resolved,
        inputs=inputs,
        pattern_limit=args.pattern_limit,
        checkpoint_subdir_override=args.checkpoint_subdir,
    )

    # 4) Run jobs (or print only with dry-run), then persist run manifest.
    summaries = run_prediction_jobs(
        jobs=jobs,
        split_rows=inputs["split_rows"],
        entity_types=inputs["entity_types"],
        id_to_label=inputs["id_to_label"],
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_generation_length=args.max_generation_length,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    run_manifest_path = write_step04_run_manifest(
        root=root,
        manifest_path=resolved["step04_manifest"],
        jobs=jobs,
        summaries=summaries,
        dry_run=args.dry_run,
    )

    # 5) Print concise completion summary.
    print_step04_summary(jobs, summaries, run_manifest_path, args.dry_run)


if __name__ == "__main__":
    main()
