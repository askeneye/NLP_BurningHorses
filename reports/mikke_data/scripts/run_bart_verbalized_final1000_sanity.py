from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
MIKKE_SCRIPT_DIR = PROJECT_ROOT / "nlp_burninghorses" / "MIKKE" / "scripts"
PATTERNS_PATH = MIKKE_SCRIPT_DIR / "patterns.yaml"

SPLIT = "k5_seed242"
PATTERNS = [f"pattern_{index:02d}" for index in range(1, 11)]
METHOD = "pet_oada_verbalized"
BART_SUFFIX = "xe_verbalized_final1000"
SOFT_ROOT = "data/interim/conll2003_soft_labels/bart_base/pet_oada_verbalized"
ENSEMBLE_ROOT = "data/interim/conll2003_ensemble_teachers/bart_base/pet_oada_verbalized_final1000"
FILTERED_ROOT = "data/interim/conll2003_ensemble_teachers/bart_base/pet_oada_verbalized_final1000_filtered"
REPORT_PATH = "reports/ensemble_tagging/k5_seed242_pet_oada_verbalized_final1000_test_metrics.json"
BERT_OUTPUT_DIR = (
    "models/bert_conll_distilled_fixed_final600_report/"
    "k5_seed242/pat_perm_bart_ensemble_to_bert_final1000/hard_argmax"
)


def run_python(script: Path, env: dict[str, str]) -> None:
    merged_env = os.environ.copy()
    merged_env.update(env)
    subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT, env=merged_env, check=True)


def source_pattern_ids() -> dict[str, str]:
    payload = yaml.safe_load(PATTERNS_PATH.read_text(encoding="utf-8")) or {}
    patterns = payload.get("patterns", [])
    return {
        f"pattern_{index:02d}": str(pattern["id"])
        for index, pattern in enumerate(patterns, start=1)
    }


def train_members() -> None:
    pattern_ids = source_pattern_ids()
    for pattern in PATTERNS:
        output_dir = (
            PROJECT_ROOT
            / "models"
            / "conll2003_kshot_seq2seq"
            / "bart_base"
            / METHOD
            / SPLIT
            / f"{pattern}_{BART_SUFFIX}"
        )
        summary_path = output_dir / "training_summary.json"
        if summary_path.exists():
            print(f"Skipping existing BART member {SPLIT} {pattern}: {summary_path}", flush=True)
            continue

        print(f"Training canonical verbalized BART final1000 {SPLIT} {pattern}", flush=True)
        run_python(
            MIKKE_SCRIPT_DIR / "run_bart_oada_xe.py",
            {
                "BART_METHOD": METHOD,
                "BART_LOSS_TYPE": "xe",
                "BART_SEED": "42",
                "BART_KSHOT_SPLIT": SPLIT,
                "BART_PATTERN": pattern,
                "BART_SOURCE_PATTERN_ID": pattern_ids[pattern],
                "BART_TRAIN_FILE": f"data/interim/conll2003_kshot_seq2seq/train/{METHOD}/{SPLIT}/{pattern}.jsonl",
                "BART_MINI_VAL_GOLD_FILE": "data/interim/conll2003_kshot_bert/mini_val.jsonl",
                "BART_VALIDATION_GOLD_FILE": "data/interim/conll2003_kshot_bert/validation.jsonl",
                "BART_OUTPUT_DIR": str(output_dir),
                "BART_MAX_STEPS": "1000",
                "BART_EVAL_STEPS": "100",
                "BART_USE_EARLY_STOPPING": "false",
                "BART_MODEL_SELECTION_STRATEGY": "final_step",
                "BART_OADA_TAU_START": "0.0",
                "BART_OADA_TAU_END": "1.0",
                "BART_OADA_TAU_WARMUP_STEPS": "1000",
                "BART_OADA_CANDIDATE_CAP": "24",
                "BART_OADA_CANDIDATE_SEED": "42",
            },
        )


def generate_hard_labels() -> None:
    for pattern in PATTERNS:
        model_dir = (
            PROJECT_ROOT
            / "models"
            / "conll2003_kshot_seq2seq"
            / "bart_base"
            / METHOD
            / SPLIT
            / f"{pattern}_{BART_SUFFIX}"
        )
        for eval_split in ["unlabeled_pool", "test"]:
            output_dir = PROJECT_ROOT / SOFT_ROOT / SPLIT / f"{pattern}_{BART_SUFFIX}" / eval_split
            if (output_dir / "hard_predictions.jsonl").exists():
                print(f"Skipping existing hard labels {SPLIT} {pattern} {eval_split}", flush=True)
                continue

            input_file = (
                f"data/interim/conll2003_kshot_bert/{SPLIT}/unlabeled_pool.jsonl"
                if eval_split == "unlabeled_pool"
                else "data/interim/conll2003_kshot_bert/test.jsonl"
            )
            print(f"Generating {eval_split} hard labels {SPLIT} {pattern}", flush=True)
            run_python(
                MIKKE_SCRIPT_DIR / "generate_bart_member_soft_labels.py",
                {
                    "SOFT_LABEL_DATASET": "conll2003",
                    "SOFT_LABEL_METHOD": METHOD,
                    "SOFT_LABEL_MODEL_FAMILY": "bart_base",
                    "SOFT_LABEL_KSHOT_SPLIT": SPLIT,
                    "SOFT_LABEL_PATTERN": pattern,
                    "SOFT_LABEL_INPUT_FILE": input_file,
                    "SOFT_LABEL_METADATA_FILE": "data/interim/conll2003_kshot_bert/metadata.json",
                    "SOFT_LABEL_MODEL_DIR": str(model_dir),
                    "SOFT_LABEL_CHECKPOINT_DIR": str(model_dir / "final_step"),
                    "SOFT_LABEL_OUTPUT_DIR": str(output_dir),
                    "SOFT_LABEL_PATTERN_SECTION": "patterns",
                    "SOFT_LABEL_INFERENCE_ORDER": "first to last",
                    "SOFT_LABEL_HARD_ONLY": "1",
                },
            )


def aggregate_and_score() -> None:
    print("Aggregating final1000 test and unlabeled-pool ensembles", flush=True)
    run_python(
        MIKKE_SCRIPT_DIR / "aggregate_bart_ensemble_votes.py",
        {
            "ENSEMBLE_TEACHER_SPLITS": SPLIT,
            "ENSEMBLE_TEACHER_PATTERNS": ",".join(PATTERNS),
            "ENSEMBLE_TEACHER_EVAL_SPLITS": "unlabeled_pool,test",
            "ENSEMBLE_TEACHER_SOURCE_SUFFIX": BART_SUFFIX,
            "ENSEMBLE_TEACHER_INPUT_ROOT": SOFT_ROOT,
            "ENSEMBLE_TEACHER_OUTPUT_ROOT": ENSEMBLE_ROOT,
        },
    )
    print("Scoring final1000 test ensemble", flush=True)
    run_python(
        MIKKE_SCRIPT_DIR / "evaluate_ensemble_hard_predictions.py",
        {
            "ENSEMBLE_EVAL_INPUT_FILE": f"{ENSEMBLE_ROOT}/{SPLIT}/test/ensemble_hard_argmax.jsonl",
            "ENSEMBLE_EVAL_OUTPUT_FILE": REPORT_PATH,
        },
    )


def filter_and_distill() -> None:
    print("Filtering final1000 ensemble by agreement", flush=True)
    run_python(
        MIKKE_SCRIPT_DIR / "filter_ensemble_teacher_subsets.py",
        {
            "FILTER_TEACHER_SPLITS": SPLIT,
            "FILTER_TEACHER_SIZES": "1000",
            "FILTER_TEACHER_INPUT_ROOT": ENSEMBLE_ROOT,
            "FILTER_TEACHER_OUTPUT_ROOT": FILTERED_ROOT,
        },
    )

    output_dir = PROJECT_ROOT / BERT_OUTPUT_DIR
    summary_path = output_dir / "training_summary.json"
    if summary_path.exists():
        print(f"Skipping existing final1000 BERT distillation: {summary_path}", flush=True)
        return

    print("Distilling final1000 agreement-filtered ensemble to BERT", flush=True)
    run_python(
        PROJECT_ROOT / "nlp_burninghorses" / "bert_conll_distill.py",
        {
            "DISTILL_SPLIT": SPLIT,
            "DISTILL_TEACHER_SPLIT": "unlabeled_pool_top_1000_final1000",
            "DISTILL_TEACHER_VARIANT": "hard_argmax",
            "DISTILL_TRAIN_FILE": f"{FILTERED_ROOT}/{SPLIT}/unlabeled_pool/top_1000/ensemble_hard_argmax.jsonl",
            "DISTILL_OUTPUT_DIR": str(output_dir),
            "DISTILL_MAX_STEPS": "600",
            "DISTILL_EVAL_EVERY": "100",
            "DISTILL_PATIENCE": "5",
            "DISTILL_EARLY_STOPPING": "false",
            "DISTILL_USE_BEST_CHECKPOINT": "false",
            "DISTILL_SEED": "42",
        },
    )


def main() -> None:
    print(
        "Canonical verbalized BART final1000 sanity check: "
        f"split={SPLIT} patterns={','.join(PATTERNS)}",
        flush=True,
    )
    train_members()
    generate_hard_labels()
    aggregate_and_score()
    filter_and_distill()
    print(f"Done. Metrics: {PROJECT_ROOT / REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
