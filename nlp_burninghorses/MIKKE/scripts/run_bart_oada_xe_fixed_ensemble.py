from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start_path: Path) -> Path:
    for path in [start_path, *start_path.parents]:
        if (path / "pyproject.toml").exists():
            return path

    raise FileNotFoundError("Could not locate project root containing pyproject.toml.")


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
RUNNER_SCRIPT = SCRIPT_DIR / "run_bart_oada_xe.py"
PATTERNS_FILE = SCRIPT_DIR / "patterns.yaml"

DEFAULT_SPLITS = ["k5_seed42", "k5_seed142", "k5_seed242"]
DEFAULT_PATTERN = "pattern_01"
DEFAULT_LOSS_TYPE = "oada_xe"
DEFAULT_MAX_STEPS = "2000"
DEFAULT_EVAL_STEPS = "100"
DEFAULT_OADA_CANDIDATE_CAP = "24"
DEFAULT_OADA_CANDIDATE_SEED = "42"
DEFAULT_SEED = "42"

FIXED_POLICY_ENV = {
    "BART_USE_EARLY_STOPPING": "false",
    "BART_MODEL_SELECTION_STRATEGY": "final_step",
    "BART_OADA_TAU_START": "0.0",
    "BART_OADA_TAU_END": "1.0",
}


def split_names_from_env() -> list[str]:
    raw_splits = os.environ.get("BART_ENSEMBLE_SPLITS")
    if raw_splits is None:
        return DEFAULT_SPLITS

    return [split.strip() for split in raw_splits.split(",") if split.strip()]


def pattern_names_from_env() -> list[str]:
    raw_patterns = os.environ.get("BART_ENSEMBLE_PATTERNS")
    if raw_patterns is None:
        return [DEFAULT_PATTERN]

    return [pattern.strip() for pattern in raw_patterns.split(",") if pattern.strip()]


def bool_from_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def source_pattern_ids() -> dict[str, str]:
    with PATTERNS_FILE.open("r", encoding="utf-8") as input_file:
        pattern_config = yaml.safe_load(input_file) or {}

    patterns = pattern_config.get("patterns", [])
    if not isinstance(patterns, list):
        raise ValueError(f"{PATTERNS_FILE} field 'patterns' must be a list.")

    pattern_ids: dict[str, str] = {}
    for index, pattern in enumerate(patterns, start=1):
        if not isinstance(pattern, dict) or not isinstance(pattern.get("id"), str):
            raise ValueError(f"{PATTERNS_FILE} pattern #{index} must define a string id.")
        pattern_ids[f"pattern_{index:02d}"] = pattern["id"]

    return pattern_ids


def policy_from_env() -> dict[str, str]:
    loss_type = os.environ.get("BART_ENSEMBLE_LOSS_TYPE", DEFAULT_LOSS_TYPE)
    max_steps = os.environ.get("BART_ENSEMBLE_MAX_STEPS", DEFAULT_MAX_STEPS)
    eval_steps = os.environ.get("BART_ENSEMBLE_EVAL_STEPS", DEFAULT_EVAL_STEPS)
    candidate_cap = os.environ.get(
        "BART_ENSEMBLE_OADA_CANDIDATE_CAP",
        DEFAULT_OADA_CANDIDATE_CAP,
    )
    candidate_seed = os.environ.get(
        "BART_ENSEMBLE_OADA_CANDIDATE_SEED",
        DEFAULT_OADA_CANDIDATE_SEED,
    )
    seed = os.environ.get("BART_ENSEMBLE_SEED", DEFAULT_SEED)
    suffix = os.environ.get("BART_ENSEMBLE_OUTPUT_SUFFIX")
    if suffix is None:
        if loss_type == "oada_xe":
            suffix = f"oada_xe_cap{candidate_cap}_fixed{max_steps}"
        elif loss_type == "xe":
            suffix = f"xe_fixed{max_steps}"
        else:
            raise ValueError("BART_ENSEMBLE_LOSS_TYPE must be one of: oada_xe, xe")

    return {
        "loss_type": loss_type,
        "max_steps": max_steps,
        "eval_steps": eval_steps,
        "candidate_cap": candidate_cap,
        "candidate_seed": candidate_seed,
        "seed": seed,
        "suffix": suffix,
    }


def train_file_for_split_pattern(split: str, pattern: str) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "interim"
        / "conll2003_kshot_seq2seq"
        / "train"
        / "pet_oada"
        / split
        / f"{pattern}.jsonl"
    )


def output_dir_for_split_pattern(split: str, pattern: str, suffix: str) -> Path:
    return (
        PROJECT_ROOT
        / "models"
        / "conll2003_kshot_seq2seq"
        / "bart_base"
        / "pet_oada"
        / split
        / f"{pattern}_{suffix}"
    )


def run_member(
    split: str,
    pattern: str,
    source_pattern_id: str,
    overwrite: bool,
    policy: dict[str, str],
) -> None:
    train_file = train_file_for_split_pattern(split, pattern)
    output_dir = output_dir_for_split_pattern(split, pattern, suffix=policy["suffix"])
    summary_path = output_dir / "training_summary.json"

    if not train_file.exists():
        raise FileNotFoundError(f"Missing train file for {split} {pattern}: {train_file}")

    if summary_path.exists() and not overwrite:
        print(f"Skipping {split} {pattern}: summary already exists at {summary_path}")
        return

    env = os.environ.copy()
    env.update(FIXED_POLICY_ENV)
    env["BART_LOSS_TYPE"] = policy["loss_type"]
    env["BART_MAX_STEPS"] = policy["max_steps"]
    env["BART_EVAL_STEPS"] = policy["eval_steps"]
    env["BART_OADA_TAU_WARMUP_STEPS"] = policy["max_steps"]
    env["BART_OADA_CANDIDATE_CAP"] = policy["candidate_cap"]
    env["BART_OADA_CANDIDATE_SEED"] = policy["candidate_seed"]
    env["BART_SEED"] = policy["seed"]
    env["BART_KSHOT_SPLIT"] = split
    env["BART_PATTERN"] = pattern
    env["BART_SOURCE_PATTERN_ID"] = source_pattern_id
    env["BART_TRAIN_FILE"] = str(train_file)
    env["BART_OUTPUT_DIR"] = str(output_dir)

    print(f"\n=== Training fixed-step {policy['loss_type']} member: {split} {pattern} ===")
    print(f"train_file={train_file}")
    print(f"output_dir={output_dir}")
    subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def main() -> None:
    splits = split_names_from_env()
    patterns = pattern_names_from_env()
    pattern_ids = source_pattern_ids()
    overwrite = bool_from_env("BART_ENSEMBLE_OVERWRITE", default=False)
    policy = policy_from_env()

    missing_pattern_ids = sorted(set(patterns).difference(pattern_ids))
    if missing_pattern_ids:
        raise ValueError(
            "No source pattern id found for: " + ", ".join(missing_pattern_ids)
        )

    print("Fixed-step BART ensemble policy")
    print(f"splits={', '.join(splits)}")
    print(f"patterns={', '.join(patterns)}")
    print(f"overwrite={overwrite}")
    print(
        " ".join(
            [
                f"loss_type={policy['loss_type']}",
                f"max_steps={policy['max_steps']}",
                f"eval_steps={policy['eval_steps']}",
                f"tau_warmup={policy['max_steps']}",
                f"cap={policy['candidate_cap']}",
                f"seed={policy['seed']}",
                f"output_suffix={policy['suffix']}",
                "model_selection=final_step",
            ]
        )
    )

    for split in splits:
        for pattern in patterns:
            run_member(
                split,
                pattern=pattern,
                source_pattern_id=pattern_ids[pattern],
                overwrite=overwrite,
                policy=policy,
            )


if __name__ == "__main__":
    main()
