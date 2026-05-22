from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts._lib.repro_io import load_yaml, project_root
from scripts._lib.step07_score_bert import (
    print_step07_summary,
    resolve_step07_paths,
    score_student_model,
    write_step07_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score final distilled BERT model on mini-val/validation/test.")
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

    # 1) Resolve paths and load latest step-06 manifest.
    resolved = resolve_step07_paths(root, config)
    step06_manifest_path = Path(resolved["step06_manifest"])
    if not step06_manifest_path.exists():
        raise FileNotFoundError(f"Missing Step-06 run manifest: {step06_manifest_path}")
    step06_manifest = json.loads(step06_manifest_path.read_text(encoding="utf-8"))
    job = step06_manifest["job"]
    model_dir = Path(job["output_dir"]) / "last_step"
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing last_step checkpoint to score: {model_dir}")

    # 2) Compute span strict metrics for all eval splits.
    metrics = score_student_model(
        base_root=Path(resolved["base_root"]),
        model_dir=model_dir,
    )

    # 3) Write report and run manifest, then print concise summary.
    report_path, manifest_path = write_step07_outputs(
        reports_root=Path(resolved["reports_root"]),
        manifest_path=Path(resolved["step07_manifest"]),
        split_name=str(resolved["split_name"]),
        teacher_variant=str(job["teacher_variant"]),
        run_tag=str(job["run_tag"]),
        model_dir=model_dir,
        metrics=metrics,
    )
    print_step07_summary(report_path, manifest_path, metrics)


if __name__ == "__main__":
    main()
