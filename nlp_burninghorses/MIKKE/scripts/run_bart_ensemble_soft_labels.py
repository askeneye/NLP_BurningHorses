from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start_path: Path) -> Path:
    for path in [start_path, *start_path.parents]:
        if (path / "pyproject.toml").exists():
            return path

    raise FileNotFoundError("Could not locate project root containing pyproject.toml.")


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
GENERATOR_SCRIPT = SCRIPT_DIR / "generate_bart_member_soft_labels.py"

DEFAULT_LOW_K_SPLITS = ["k5_seed42", "k5_seed142", "k5_seed242", "k10_seed42", "k10_seed142", "k10_seed242"]
DEFAULT_HIGH_K_SPLITS = ["k20_seed42", "k20_seed142", "k20_seed242", "k50_seed42", "k50_seed142", "k50_seed242"]
DEFAULT_LOW_K_PATTERNS = [f"pattern_{index:02d}" for index in range(1, 11)]
DEFAULT_HIGH_K_PATTERNS = [f"pattern_{index:02d}" for index in range(1, 6)]
DEFAULT_OUTPUT_SUFFIX = "xe_improved_patterns_fixed2000"
DEFAULT_CHECKPOINT_SUBDIR = "final_step"
DEFAULT_EVAL_SPLITS = {
    "mini_val": "data/interim/conll2003_kshot_bert/mini_val.jsonl",
    "validation": "data/interim/conll2003_kshot_bert/validation.jsonl",
    "unlabeled_pool": "data/interim/conll2003_kshot_bert/{split}/unlabeled_pool.jsonl",
}


def csv_from_env(name: str, default: list[str]) -> list[str]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def bool_from_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def split_plan() -> list[tuple[list[str], list[str]]]:
    all_splits = os.environ.get("SOFT_LABEL_ENSEMBLE_SPLITS")
    all_patterns = os.environ.get("SOFT_LABEL_ENSEMBLE_PATTERNS")
    if all_splits is not None or all_patterns is not None:
        return [
            (
                csv_from_env("SOFT_LABEL_ENSEMBLE_SPLITS", DEFAULT_LOW_K_SPLITS + DEFAULT_HIGH_K_SPLITS),
                csv_from_env("SOFT_LABEL_ENSEMBLE_PATTERNS", DEFAULT_LOW_K_PATTERNS),
            )
        ]

    return [
        (DEFAULT_LOW_K_SPLITS, DEFAULT_LOW_K_PATTERNS),
        (DEFAULT_HIGH_K_SPLITS, DEFAULT_HIGH_K_PATTERNS),
    ]


def eval_split_files() -> dict[str, Path]:
    requested_splits = csv_from_env("SOFT_LABEL_EVAL_SPLITS", list(DEFAULT_EVAL_SPLITS))
    split_files = {
        "mini_val": os.environ.get("SOFT_LABEL_MINI_VAL_FILE", DEFAULT_EVAL_SPLITS["mini_val"]),
        "validation": os.environ.get("SOFT_LABEL_VALIDATION_FILE", DEFAULT_EVAL_SPLITS["validation"]),
        "unlabeled_pool": os.environ.get(
            "SOFT_LABEL_UNLABELED_POOL_FILE",
            DEFAULT_EVAL_SPLITS["unlabeled_pool"],
        ),
    }

    missing = sorted(set(requested_splits).difference(split_files))
    if missing:
        raise ValueError("Unsupported SOFT_LABEL_EVAL_SPLITS entries: " + ", ".join(missing))

    return {split: PROJECT_ROOT / split_files[split] for split in requested_splits}


def model_dir_for(split: str, pattern: str, suffix: str) -> Path:
    return (
        PROJECT_ROOT
        / "models"
        / "conll2003_kshot_seq2seq"
        / "bart_base"
        / "pet_oada"
        / split
        / f"{pattern}_{suffix}"
    )


def output_dir_for(split: str, pattern: str, suffix: str, eval_split: str) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "interim"
        / "conll2003_soft_labels"
        / "bart_base"
        / "pet_oada"
        / split
        / f"{pattern}_{suffix}"
        / eval_split
    )


def should_skip(output_dir: Path, overwrite: bool, hard_only: bool) -> bool:
    if overwrite:
        return False
    required_outputs = [
        output_dir / "hard_predictions.jsonl",
        output_dir / "manifest.json",
    ]
    if not hard_only:
        required_outputs.append(output_dir / "soft_labels.jsonl")
    return all(path.exists() for path in required_outputs)


def run_member_split(
    split: str,
    pattern: str,
    eval_split: str,
    input_file: Path,
    suffix: str,
    checkpoint_subdir: str,
    overwrite: bool,
    hard_only: bool,
) -> None:
    model_dir = model_dir_for(split, pattern, suffix)
    checkpoint_dir = model_dir / checkpoint_subdir
    output_dir = output_dir_for(split, pattern, suffix, eval_split)

    if not (model_dir / "training_summary.json").exists():
        raise FileNotFoundError(f"Missing training summary: {model_dir / 'training_summary.json'}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")
    if not input_file.exists():
        raise FileNotFoundError(f"Missing input file for {eval_split}: {input_file}")
    if should_skip(output_dir, overwrite, hard_only):
        print(f"Skipping {split} {pattern} {eval_split}: outputs already exist at {output_dir}")
        return

    env = os.environ.copy()
    env.update(
        {
            "SOFT_LABEL_DATASET": "conll2003",
            "SOFT_LABEL_METHOD": "pet_oada",
            "SOFT_LABEL_MODEL_FAMILY": "bart_base",
            "SOFT_LABEL_KSHOT_SPLIT": split,
            "SOFT_LABEL_PATTERN": pattern,
            "SOFT_LABEL_INPUT_FILE": str(input_file),
            "SOFT_LABEL_METADATA_FILE": str(PROJECT_ROOT / "data/interim/conll2003_kshot_bert/metadata.json"),
            "SOFT_LABEL_MODEL_DIR": str(model_dir),
            "SOFT_LABEL_CHECKPOINT_DIR": str(checkpoint_dir),
            "SOFT_LABEL_OUTPUT_DIR": str(output_dir),
            "SOFT_LABEL_HARD_ONLY": "1" if hard_only else "0",
        }
    )

    print(f"\n=== Soft labels: {split} {pattern} {eval_split} ===")
    print(f"model_dir={model_dir}")
    print(f"input_file={input_file}")
    print(f"output_dir={output_dir}")
    subprocess.run([sys.executable, str(GENERATOR_SCRIPT)], cwd=PROJECT_ROOT, env=env, check=True)


def main() -> None:
    suffix = os.environ.get("SOFT_LABEL_ENSEMBLE_OUTPUT_SUFFIX", DEFAULT_OUTPUT_SUFFIX)
    checkpoint_subdir = os.environ.get("SOFT_LABEL_CHECKPOINT_SUBDIR", DEFAULT_CHECKPOINT_SUBDIR)
    overwrite = bool_from_env("SOFT_LABEL_OVERWRITE", default=False)
    hard_only = bool_from_env("SOFT_LABEL_HARD_ONLY", default=False)
    splits_and_patterns = split_plan()
    split_files = eval_split_files()

    print("BART ensemble soft-label extraction")
    print(f"suffix={suffix}")
    print(f"checkpoint_subdir={checkpoint_subdir}")
    print(f"eval_splits={', '.join(split_files)}")
    print(f"overwrite={overwrite}")
    print(f"hard_only={hard_only}")

    for splits, patterns in splits_and_patterns:
        print(f"planned_splits={', '.join(splits)}")
        print(f"planned_patterns={', '.join(patterns)}")
        for split in splits:
            for pattern in patterns:
                for eval_split, input_file in split_files.items():
                    resolved_input_file = Path(str(input_file).format(split=split))
                    run_member_split(
                        split=split,
                        pattern=pattern,
                        eval_split=eval_split,
                        input_file=resolved_input_file,
                        suffix=suffix,
                        checkpoint_subdir=checkpoint_subdir,
                        overwrite=overwrite,
                        hard_only=hard_only,
                    )


if __name__ == "__main__":
    main()
