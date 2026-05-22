from __future__ import annotations

import argparse
from pathlib import Path

from scripts._lib.repro_io import load_yaml, project_root
from scripts._lib.step06_train_bert import (
    build_distill_job,
    print_step06_summary,
    resolve_step06_paths,
    run_distillation,
    write_step06_run_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train distilled BERT student from ensemble labels.")
    parser.add_argument(
        "--config",
        default="scripts/config/reproduction_default.yaml",
        help="Path to reproduction YAML config.",
    )
    parser.add_argument(
        "--max-steps-override",
        type=int,
        default=None,
        help="Optionally override student max steps for smoke runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print distillation plan without running training.",
    )
    parser.add_argument(
        "--train-file-override",
        default=None,
        help="Optional prefiltered label file to train from.",
    )
    parser.add_argument(
        "--output-name-override",
        default=None,
        help="Optional student output directory name under reproduction/models/student/<split>/.",
    )
    parser.add_argument(
        "--run-tag-override",
        default=None,
        help="Optional run tag for reports/manifests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)

    # 1) Resolve paths and build concrete distillation job.
    resolved = resolve_step06_paths(root, config)
    train_file_override = Path(args.train_file_override) if args.train_file_override else None
    job = build_distill_job(
        config=config,
        resolved_paths=resolved,
        max_steps_override=args.max_steps_override,
        train_file_override=train_file_override,
        output_name_override=args.output_name_override,
        run_tag_override=args.run_tag_override,
    )

    # 2) Run distillation (or print only with dry-run).
    summary = run_distillation(job, dry_run=args.dry_run)

    # 3) Persist run manifest and print concise summary.
    run_manifest_path = write_step06_run_manifest(
        manifest_path=resolved["step06_manifest"],
        job=job,
        summary=summary,
        dry_run=args.dry_run,
    )
    print_step06_summary(summary, run_manifest_path, args.dry_run)


if __name__ == "__main__":
    main()
