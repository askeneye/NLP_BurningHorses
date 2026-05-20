"""Analyze BERT k-shot baseline early-stopping behavior vs fixed-600 runs."""
from __future__ import annotations

import csv
import statistics as stats
from collections import defaultdict
from pathlib import Path

BATCH = 16
EVAL_EVERY = 15
PATIENCE = 5
MIN_BEFORE = 150


def load_test_rows(path: Path) -> dict:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if record["Eval_Split"] != "test":
                continue
            key = (int(record["K_shot"]), int(record["Split_Seed"]))
            rows[key] = {
                "split": record["Split_Name"],
                "k": int(record["K_shot"]),
                "seed": int(record["Split_Seed"]),
                "n_train": int(record["N_Train_Sentences"]),
                "max_steps": int(record["Max_Train_Steps"]),
                "train_steps": int(record["Train_Steps"]),
                "best_step": int(record["Best_Span_F1_Step"]),
                "early_stopped": int(record["Early_Stopped"]),
                "test_f1": float(record["Span_Strict_F1"]),
            }
    return rows


def steps_per_epoch(n_train: int) -> int:
    return max(1, (n_train + BATCH - 1) // BATCH)


def epoch_equiv(step: int, n_train: int) -> float:
    return step / steps_per_epoch(n_train)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    early_path = root / "results/bert_conll2003_standard_report/BERT_kshot_standard_report.csv"
    fixed_path = (
        root / "results/bert_conll2003_standard_report_fixed600/BERT_kshot_standard_report_fixed600.csv"
    )

    early = load_test_rows(early_path)
    fixed = load_test_rows(fixed_path) if fixed_path.exists() else {}

    runs = []
    for key in sorted(early):
        row = early[key]
        spe = steps_per_epoch(row["n_train"])
        runs.append(
            {
                **row,
                "steps_per_epoch": spe,
                "epochs_at_best": epoch_equiv(row["best_step"], row["n_train"]),
                "epochs_at_stop": epoch_equiv(row["train_steps"], row["n_train"]),
                "budget_used_pct": 100 * row["train_steps"] / row["max_steps"],
                "stop_after_best_steps": row["train_steps"] - row["best_step"],
                "fixed600_test_f1": fixed.get(key, {}).get("test_f1"),
                "fixed600_best_step": fixed.get(key, {}).get("best_step"),
            }
        )

    by_k: dict[int, list] = defaultdict(list)
    for row in runs:
        by_k[row["k"]].append(row)

    print("=== BERT baseline early-stopping diagnostics (12 test rows) ===")
    print(
        f"Policy: eval_every={EVAL_EVERY}, patience={PATIENCE}, "
        f"min_steps={MIN_BEFORE}, max_steps=1000"
    )
    print(f"Expected stop margin after last improvement: {PATIENCE * EVAL_EVERY} steps\n")
    header = (
        f"{'k':>3} {'n_tr':>5} {'best':>6} {'stop':>6} {'used%':>6} "
        f"{'ep@best':>8} {'ep@stop':>8} {'margin':>6} {'testF1':>7}"
    )
    print(header)
    for k in sorted(by_k):
        group = by_k[k]
        print(
            f"{k:3d} "
            f"{stats.mean(r['n_train'] for r in group):5.0f} "
            f"{stats.median(r['best_step'] for r in group):6.0f} "
            f"{stats.median(r['train_steps'] for r in group):6.0f} "
            f"{stats.median(r['budget_used_pct'] for r in group):5.1f} "
            f"{stats.median(r['epochs_at_best'] for r in group):8.1f} "
            f"{stats.median(r['epochs_at_stop'] for r in group):8.1f} "
            f"{stats.median(r['stop_after_best_steps'] for r in group):6.0f} "
            f"{stats.mean(r['test_f1'] for r in group) * 100:6.2f}"
        )

    print("\n=== Per-run detail ===")
    for row in runs:
        delta = ""
        if row["fixed600_test_f1"] is not None:
            diff = (row["fixed600_test_f1"] - row["test_f1"]) * 100
            delta = (
                f" fixed600={row['fixed600_test_f1'] * 100:.2f} "
                f"({diff:+.2f} vs early-stop)"
            )
        print(
            f"{row['split']:14s} n={row['n_train']:2d} best={row['best_step']:3d} "
            f"stop={row['train_steps']:3d} margin={row['stop_after_best_steps']:3d} "
            f"ep@best={row['epochs_at_best']:6.1f} test={row['test_f1'] * 100:5.2f}{delta}"
        )

    pairs = [row for row in runs if row["fixed600_test_f1"] is not None]
    if pairs:
        print("\n=== Early-stop vs fixed-600 final checkpoint (paired test F1) ===")
        for k in sorted(by_k):
            group = [row for row in pairs if row["k"] == k]
            if not group:
                continue
            early_mean = stats.mean(row["test_f1"] for row in group) * 100
            fixed_mean = stats.mean(row["fixed600_test_f1"] for row in group) * 100
            print(
                f"k={k:2d}: early-stop {early_mean:.2f} | fixed-600 {fixed_mean:.2f} | "
                f"delta {fixed_mean - early_mean:+.2f} (n={len(group)})"
            )
        all_early = stats.mean(row["test_f1"] for row in pairs) * 100
        all_fixed = stats.mean(row["fixed600_test_f1"] for row in pairs) * 100
        print(
            f"ALL paired ({len(pairs)}): early {all_early:.2f} | fixed600 {all_fixed:.2f} | "
            f"delta {all_fixed - all_early:+.2f}"
        )

    print("\n=== Suggested k-dependent upper budgets (p75 best_step, early-stop era) ===")
    for k in sorted(by_k):
        bests = sorted(row["best_step"] for row in by_k[k])
        p75 = bests[int(0.75 * (len(bests) - 1))]
        cap = int((p75 + 49) // 50 * 50)
        print(f"k={k:2d}: best steps {bests} -> p75={p75}, suggest cap ~{cap}")

    out = root / "reports/mikke_data/data/bert_baseline_stopping_analysis.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "k",
        "seed",
        "n_train",
        "max_steps",
        "train_steps",
        "best_step",
        "early_stopped",
        "steps_per_epoch",
        "epochs_at_best",
        "epochs_at_stop",
        "budget_used_pct",
        "stop_after_best_steps",
        "test_f1_early_stop",
        "test_f1_fixed600",
        "test_f1_delta_fixed_minus_early",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in runs:
            writer.writerow(
                {
                    "split": row["split"],
                    "k": row["k"],
                    "seed": row["seed"],
                    "n_train": row["n_train"],
                    "max_steps": row["max_steps"],
                    "train_steps": row["train_steps"],
                    "best_step": row["best_step"],
                    "early_stopped": row["early_stopped"],
                    "steps_per_epoch": row["steps_per_epoch"],
                    "epochs_at_best": f"{row['epochs_at_best']:.2f}",
                    "epochs_at_stop": f"{row['epochs_at_stop']:.2f}",
                    "budget_used_pct": f"{row['budget_used_pct']:.1f}",
                    "stop_after_best_steps": row["stop_after_best_steps"],
                    "test_f1_early_stop": f"{row['test_f1']:.6f}",
                    "test_f1_fixed600": (
                        ""
                        if row["fixed600_test_f1"] is None
                        else f"{row['fixed600_test_f1']:.6f}"
                    ),
                    "test_f1_delta_fixed_minus_early": (
                        ""
                        if row["fixed600_test_f1"] is None
                        else f"{row['fixed600_test_f1'] - row['test_f1']:.6f}"
                    ),
                }
            )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
