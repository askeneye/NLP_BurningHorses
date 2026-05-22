from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import time

from scripts._lib.repro_io import load_yaml
from scripts.run_single_ensemble_to_bert import (
    _print_celebration_card,
    _read_expected_test_strict_f1,
    _read_observed_test_strict_f1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single-teacher to BERT reproduction pipeline.")
    parser.add_argument(
        "--config",
        default="scripts/config/reproduction_default.yaml",
        help="Path to reproduction YAML config.",
    )
    parser.add_argument(
        "--start-at",
        default="00",
        choices=["00", "01", "02", "03", "04", "05", "06", "07"],
        help="Start pipeline from this step id.",
    )
    parser.add_argument(
        "--no-celebrate",
        action="store_true",
        help="Skip completion celebration card and animation.",
    )
    return parser.parse_args()


def _run(cmd: list[str], *, step_id: str, heartbeat_s: int = 30) -> None:
    print(f"[step {step_id}] {' '.join(cmd)}", flush=True)
    process = subprocess.Popen(cmd)
    started = time.time()
    next_heartbeat = started + heartbeat_s
    while True:
        return_code = process.poll()
        if return_code is not None:
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
            elapsed = int(time.time() - started)
            print(f"[step {step_id}] done in {elapsed}s", flush=True)
            return
        now = time.time()
        if now >= next_heartbeat:
            elapsed = int(now - started)
            minutes, seconds = divmod(elapsed, 60)
            print(f"[step {step_id}] alive {minutes:02d}:{seconds:02d}", flush=True)
            next_heartbeat = now + heartbeat_s
        time.sleep(1.0)


def main() -> None:
    args = parse_args()
    py = sys.executable
    root = Path(__file__).resolve().parents[1]
    config = load_yaml(root / args.config)
    split_name = str(config.get("default_profile", {}).get("split_name", "k5_seed242"))
    single_teacher_label_file = (
        f"reproduction/data/interim/{split_name}/single_teacher_labels/"
        "unlabeled_pool/top_1000/hard_labels.jsonl"
    )

    steps: list[tuple[str, list[str]]] = [
        ("00", [py, "scripts/00_prepare_conll2003.py", "--config", args.config]),
        ("01", [py, "scripts/01_make_kshot_split.py", "--config", args.config]),
        ("02", [py, "scripts/02_build_bart_teacher_data.py", "--config", args.config]),
        ("03", [py, "scripts/03_train_bart_teachers.py", "--config", args.config, "--pattern-limit", "1"]),
        (
            "04",
            [
                py,
                "scripts/04_predict_with_bart_teachers.py",
                "--config",
                args.config,
                "--pattern-limit",
                "1",
                "--overwrite",
            ],
        ),
        ("05", [py, "scripts/05_prepare_single_teacher_labels.py", "--config", args.config, "--pattern-limit", "1"]),
        (
            "06",
            [
                py,
                "scripts/06_train_bert_student.py",
                "--config",
                args.config,
                "--train-file-override",
                single_teacher_label_file,
                "--output-name-override",
                "single_teacher_hard_argmax",
                "--run-tag-override",
                "single_teacher",
            ],
        ),
        ("07", [py, "scripts/07_score_final_model.py", "--config", args.config]),
    ]

    start_index = next(index for index, (step_id, _) in enumerate(steps) if step_id == args.start_at)
    for step_id, command in steps[start_index:]:
        if step_id == "06":
            print("[step 06] Note: BERT may print UNEXPECTED/MISSING keys while loading base weights; this is expected.", flush=True)
        _run(command, step_id=step_id)

    observed_test_strict_f1 = _read_observed_test_strict_f1(root, config, step_started=args.start_at)
    expected_test_strict_f1 = _read_expected_test_strict_f1(config, "single_teacher_to_bert")

    if args.no_celebrate:
        print("Single-teacher reproduction completed.")
        if observed_test_strict_f1 is not None:
            print(f"test_strict_f1={observed_test_strict_f1:.4f}")
    else:
        _print_celebration_card(
            smoke=False,
            observed_test_strict_f1=observed_test_strict_f1,
            expected_test_strict_f1=expected_test_strict_f1,
        )


if __name__ == "__main__":
    main()
