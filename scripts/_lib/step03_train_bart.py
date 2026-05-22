from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts._lib.bart_training import BartTrainingJob, train_bart_job
from scripts._lib.repro_io import load_yaml


def resolve_step03_paths(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    split_name = str(config.get("default_profile", {}).get("split_name", "k5_seed242"))
    paths = config.get("paths", {})
    interim_root = root / str(paths.get("interim_root", "reproduction/data/interim"))
    models_root = root / str(paths.get("models_root", "reproduction/models"))
    logs_root = root / str(paths.get("logs_root", "reproduction/logs"))
    split_root = interim_root / split_name
    train_manifest = split_root / "teacher_data" / "train" / "manifest.json"
    mini_val_file = split_root / "teacher_data" / "inference" / "mini_val.jsonl"
    return {
        "split_name": split_name,
        "split_root": split_root,
        "interim_root": interim_root,
        "models_root": models_root,
        "logs_root": logs_root,
        "train_manifest": train_manifest,
        "mini_val_file": mini_val_file,
    }


def load_training_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing train manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def select_pattern_ids(config: dict[str, Any], manifest: dict[str, Any], pattern_limit: int | None) -> list[str]:
    preferred = list(config.get("teacher_ensemble", {}).get("pattern_ids", []))
    available = list(manifest.keys())
    selected = [pattern_id for pattern_id in preferred if pattern_id in available] if preferred else available
    if pattern_limit is not None:
        selected = selected[:pattern_limit]
    return selected


def build_training_jobs(
    *,
    root: Path,
    config: dict[str, Any],
    resolved_paths: dict[str, Any],
    manifest: dict[str, Any],
    pattern_ids: list[str],
    max_steps_override: int | None = None,
) -> list[BartTrainingJob]:
    teacher_cfg = config.get("teacher_ensemble", {})
    train_cfg = teacher_cfg.get("train", {})
    model_name = str(teacher_cfg.get("model_name", "facebook/bart-base"))
    split_name = str(resolved_paths["split_name"])
    default_max_steps = int(train_cfg.get("max_steps", 2000))
    jobs: list[BartTrainingJob] = []

    for pattern_id in pattern_ids:
        pattern_meta = manifest[pattern_id]
        train_file = root / str(pattern_meta["output_file"])
        source_pattern_id = str(pattern_meta["source_pattern_id"])
        max_steps = int(max_steps_override) if max_steps_override is not None else default_max_steps
        run_tag = "promoted" if max_steps == default_max_steps else f"smoke_s{max_steps}"
        output_pattern_dir = f"{pattern_id}__{run_tag}" if run_tag != "promoted" else pattern_id
        output_dir = (
            Path(resolved_paths["models_root"])
            / "teachers"
            / split_name
            / output_pattern_dir
        )
        jobs.append(
            BartTrainingJob(
                split_name=split_name,
                pattern_id=pattern_id,
                source_pattern_id=source_pattern_id,
                model_name=model_name,
                train_file=train_file,
                mini_val_file=Path(resolved_paths["mini_val_file"]),
                output_dir=output_dir,
                max_source_length=128,
                max_target_length=128,
                train_batch_size=int(train_cfg.get("batch_size", 8)),
                eval_batch_size=int(train_cfg.get("eval_batch_size", 32)),
                learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
                max_steps=max_steps,
                eval_steps=int(train_cfg.get("eval_steps", 100)),
                seed=42,
                run_tag=run_tag,
            )
        )
    return jobs


def run_training_jobs(jobs: list[BartTrainingJob], dry_run: bool) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for job in jobs:
        print(
            f"train_job split={job.split_name} pattern={job.pattern_id} "
            f"steps={job.max_steps} output={job.output_dir}"
        )
        if dry_run:
            continue
        summaries.append(train_bart_job(job))
    return summaries


def write_step03_run_manifest(logs_root: Path, jobs: list[BartTrainingJob], summaries: list[dict[str, Any]]) -> Path:
    logs_root.mkdir(parents=True, exist_ok=True)
    manifest_path = logs_root / "step03_train_bart_manifest.json"
    payload = {
        "jobs": [
            {
                "split_name": job.split_name,
                "pattern_id": job.pattern_id,
                "source_pattern_id": job.source_pattern_id,
                "run_tag": job.run_tag,
                "train_file": str(job.train_file),
                "mini_val_file": str(job.mini_val_file),
                "output_dir": str(job.output_dir),
                "max_steps": job.max_steps,
                "eval_steps": job.eval_steps,
                "train_batch_size": job.train_batch_size,
            }
            for job in jobs
        ],
        "completed_summaries": [summary["output_dir"] for summary in summaries],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def print_step03_summary(
    jobs: list[BartTrainingJob],
    summaries: list[dict[str, Any]],
    run_manifest_path: Path,
    dry_run: bool,
) -> None:
    print("Prepared BART teacher training jobs")
    print(f"job_count={len(jobs)} dry_run={dry_run}")
    if summaries:
        for summary in summaries:
            experiment = summary["experiment"]
            print(
                f"finished pattern={experiment['pattern']} "
                f"tag={experiment.get('run_tag', 'promoted')} "
                f"steps={summary['steps_trained']} "
                f"avg_loss={summary['avg_loss']:.4f} "
                f"best_mini_val_loss={summary['best_mini_val_loss']}"
            )
    print(f"run_manifest={run_manifest_path}")


def run_step03_train_bart(
    root: Path,
    config_path: Path,
    *,
    pattern_limit: int | None = None,
    max_steps_override: int | None = None,
    dry_run: bool = False,
) -> None:
    config = load_yaml(config_path)
    resolved = resolve_step03_paths(root, config)
    manifest = load_training_manifest(Path(resolved["train_manifest"]))
    pattern_ids = select_pattern_ids(config, manifest, pattern_limit=pattern_limit)
    jobs = build_training_jobs(
        root=root,
        config=config,
        resolved_paths=resolved,
        manifest=manifest,
        pattern_ids=pattern_ids,
        max_steps_override=max_steps_override,
    )
    summaries = run_training_jobs(jobs, dry_run=dry_run)
    run_manifest_path = write_step03_run_manifest(Path(resolved["logs_root"]), jobs, summaries)
    print_step03_summary(jobs, summaries, run_manifest_path, dry_run)
