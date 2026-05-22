from __future__ import annotations

import argparse

from scripts._lib.repro_io import load_yaml, project_root
from scripts._lib.step05_aggregate_ensemble import (
    build_aggregation_plan,
    load_step05_inputs,
    parse_eval_splits,
    print_step05_summary,
    resolve_step05_paths,
    run_aggregation,
    write_step05_run_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate BART teacher hard predictions into ensemble teachers.")
    parser.add_argument(
        "--config",
        default="scripts/config/reproduction_default.yaml",
        help="Path to reproduction YAML config.",
    )
    parser.add_argument(
        "--eval-splits",
        default="unlabeled_pool",
        help="Comma-separated split names from {unlabeled_pool,mini_val,test,validation}.",
    )
    parser.add_argument(
        "--pattern-limit",
        type=int,
        default=None,
        help="Optionally limit number of teacher patterns used for aggregation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature override for vote_temp output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print aggregation plan without writing outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)

    # 1) Resolve paths and parse requested eval splits.
    resolved = resolve_step05_paths(root, config)
    eval_splits = parse_eval_splits(args.eval_splits)
    temperature = (
        float(args.temperature)
        if args.temperature is not None
        else float(config.get("teacher_outputs", {}).get("temperature", 2.0))
    )

    # 2) Load Step-04 manifest and label metadata.
    inputs = load_step05_inputs(
        step04_manifest_path=resolved["step04_manifest"],
        metadata_path=resolved["kshot_metadata"],
    )

    # 3) Build aggregation plan from available Step-04 member outputs.
    plan = build_aggregation_plan(
        resolved_paths=resolved,
        inputs=inputs,
        eval_splits=eval_splits,
        pattern_limit=args.pattern_limit,
    )

    # 4) Run aggregation (or print only with dry-run).
    summaries = run_aggregation(
        root=root,
        resolved_paths=resolved,
        plan=plan,
        label_list=inputs["label_list"],
        temperature=temperature,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    # 5) Persist run manifest and print summary.
    run_manifest_path = write_step05_run_manifest(
        manifest_path=resolved["step05_manifest"],
        split_name=str(resolved["split_name"]),
        plan=plan,
        summaries=summaries,
        temperature=temperature,
        dry_run=args.dry_run,
    )
    print_step05_summary(summaries, run_manifest_path, args.dry_run)


if __name__ == "__main__":
    main()
