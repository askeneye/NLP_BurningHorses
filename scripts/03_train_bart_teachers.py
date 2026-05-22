from __future__ import annotations

import argparse

from scripts._lib.repro_io import load_yaml, project_root
from scripts._lib.step03_train_bart import (
    build_training_jobs,
    load_training_manifest,
    print_step03_summary,
    resolve_step03_paths,
    run_training_jobs,
    select_pattern_ids,
    write_step03_run_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BART teacher ensemble members for reproduction.")
    parser.add_argument(
        "--config",
        default="scripts/config/reproduction_default.yaml",
        help="Path to reproduction YAML config.",
    )
    parser.add_argument(
        "--pattern-limit",
        type=int,
        default=None,
        help="Optionally limit number of patterns to train.",
    )
    parser.add_argument(
        "--max-steps-override",
        type=int,
        default=None,
        help="Optionally override training max steps for smoke runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print training plan without running training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)

    # 1) Resolve paths and load Step-02 train manifest.
    resolved = resolve_step03_paths(root, config)
    train_manifest = load_training_manifest(resolved["train_manifest"])

    # 2) Select ensemble pattern IDs for this run.
    pattern_ids = select_pattern_ids(config, train_manifest, pattern_limit=args.pattern_limit)

    # 3) Build concrete BART training jobs.
    jobs = build_training_jobs(
        root=root,
        config=config,
        resolved_paths=resolved,
        manifest=train_manifest,
        pattern_ids=pattern_ids,
        max_steps_override=args.max_steps_override,
    )

    # 4) Run jobs (or print only with dry-run), then persist run manifest.
    summaries = run_training_jobs(jobs, dry_run=args.dry_run)
    run_manifest_path = write_step03_run_manifest(resolved["logs_root"], jobs, summaries)

    # 5) Print concise completion summary.
    print_step03_summary(jobs, summaries, run_manifest_path, args.dry_run)


if __name__ == "__main__":
    main()
