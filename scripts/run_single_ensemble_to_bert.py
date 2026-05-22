from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts._lib.repro_io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full promoted pipeline with one command.")
    parser.add_argument(
        "--config",
        default="scripts/config/reproduction_default.yaml",
        help="Path to reproduction YAML config.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke settings with explicit smoke tags.",
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


def _supports_text(text: str) -> bool:
    encoding = sys.stdout.encoding or "utf-8"
    try:
        text.encode(encoding)
        return True
    except UnicodeEncodeError:
        return False


def _read_observed_test_strict_f1(root: Path, config: dict, *, step_started: str) -> float | None:
    if step_started > "07":
        return None
    logs_root = root / str(config.get("paths", {}).get("logs_root", "reproduction/logs"))
    manifest_path = logs_root / "step07_score_bert_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report_path = Path(str(manifest.get("report_file", "")))
        if not report_path.exists():
            return None
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return float(report["metrics"]["test"]["Span_Strict_F1"])
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _read_expected_test_strict_f1(config: dict, run_key: str) -> float | None:
    expected = config.get("expected_metrics", {}).get(run_key, {}).get("test_span_strict_f1")
    if expected is None:
        return None
    try:
        return float(expected)
    except (TypeError, ValueError):
        return None


def _animate_horse_back_and_forth_unicode(track_width: int = 54, frame_delay_s: float = 0.09) -> None:
    fire_frames = ("🔥", "🔥🔥", "🔥🔥🔥")
    max_offset = track_width - len("🐎💨🔥🔥🔥")
    sys.stdout.write("\x1b[?25l")
    sys.stdout.flush()
    while True:
        for position in range(max_offset, -1, -1):
            fire = fire_frames[position % len(fire_frames)]
            frame = (" " * position + f"🐎💨{fire}").ljust(track_width)
            sys.stdout.write("\x1b[2A\r" + frame + "\x1b[2B\r")
            sys.stdout.flush()
            time.sleep(frame_delay_s)


def _animate_horse_back_and_forth_ascii(track_width: int = 54, frame_delay_s: float = 0.09) -> None:
    fire_frames = ("*", "+", "x")
    max_offset = track_width - len(">>===> ~~ xxx")
    sys.stdout.write("\x1b[?25l")
    sys.stdout.flush()
    while True:
        for position in range(max_offset, -1, -1):
            fire = fire_frames[position % len(fire_frames)]
            frame = (" " * position + f"<===<< ~~ {fire}{fire}{fire}").ljust(track_width)
            sys.stdout.write("\x1b[2A\r" + frame + "\x1b[2B\r")
            sys.stdout.flush()
            time.sleep(frame_delay_s)


def _print_celebration_card(
    smoke: bool,
    *,
    observed_test_strict_f1: float | None = None,
    expected_test_strict_f1: float | None = None,
) -> None:
    use_unicode = _supports_text("✅🐎🎉🔥🙂")
    border_width = 54
    print("=" * border_width)
    print("Reproduction succes!")
    print("")
    print("        __/|__|\\__")
    print("       /  ^    ^  \\    Fire Horses run complete")
    print("      /___\\_--_/___\\")
    print("         /_/  \\_\\")
    print("")
    observed_text = "n/a" if observed_test_strict_f1 is None else f"{observed_test_strict_f1:.4f}"
    expected_text = "n/a" if expected_test_strict_f1 is None else f"{expected_test_strict_f1:.4f}"
    print(f"{'Observed test F1':>20} : {observed_text:>7}")
    print(f"{'Expected test F1':>20} : {expected_text:>7}")
    print("    Ctrl+C to quit")
    print(" " * border_width)
    print("=" * border_width)
    try:
        if use_unicode:
            _animate_horse_back_and_forth_unicode()
        else:
            _animate_horse_back_and_forth_ascii()
    except KeyboardInterrupt:
        sys.stdout.write("\x1b[?25h")
        sys.stdout.write("\x1b[2A\r" + " " * border_width + "\x1b[2B\r")
        sys.stdout.flush()
        print("")


def main() -> None:
    args = parse_args()
    py = sys.executable
    root = Path(__file__).resolve().parents[1]
    config = load_yaml(root / args.config)

    steps: list[tuple[str, list[str]]] = [
        ("00", [py, "scripts/00_prepare_conll2003.py", "--config", args.config]),
        ("01", [py, "scripts/01_make_kshot_split.py", "--config", args.config]),
        ("02", [py, "scripts/02_build_bart_teacher_data.py", "--config", args.config]),
        ("03", [py, "scripts/03_train_bart_teachers.py", "--config", args.config]),
        ("04", [py, "scripts/04_predict_with_bart_teachers.py", "--config", args.config]),
        ("05", [py, "scripts/05_aggregate_teacher_ensemble.py", "--config", args.config]),
        ("06", [py, "scripts/06_train_bert_student.py", "--config", args.config]),
        ("07", [py, "scripts/07_score_final_model.py", "--config", args.config]),
    ]

    if args.smoke:
        for step_id, command in steps:
            if step_id == "03":
                command.extend(["--pattern-limit", "1", "--max-steps-override", "1"])
            elif step_id == "04":
                command.extend(["--pattern-limit", "1", "--eval-splits", "unlabeled_pool"])
            elif step_id == "05":
                command.extend(["--pattern-limit", "1", "--eval-splits", "unlabeled_pool"])
            elif step_id == "06":
                command.extend(["--max-steps-override", "20"])

    start_index = next(index for index, (step_id, _) in enumerate(steps) if step_id == args.start_at)
    for step_id, command in steps[start_index:]:
        if step_id == "06":
            print("[step 06] Note: BERT may print UNEXPECTED/MISSING keys while loading base weights; this is expected.", flush=True)
        _run(command, step_id=step_id)

    observed_test_strict_f1 = _read_observed_test_strict_f1(root, config, step_started=args.start_at)
    expected_test_strict_f1 = _read_expected_test_strict_f1(config, "single_ensemble_to_bert")

    if args.no_celebrate:
        print(f"Reproduction completed (smoke={args.smoke})")
        if observed_test_strict_f1 is not None:
            print(f"test_strict_f1={observed_test_strict_f1:.4f}")
    else:
        _print_celebration_card(
            smoke=args.smoke,
            observed_test_strict_f1=observed_test_strict_f1,
            expected_test_strict_f1=expected_test_strict_f1,
        )


if __name__ == "__main__":
    main()
