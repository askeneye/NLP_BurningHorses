"""Replace BERT baseline rows in fixed-600 final ablation CSVs with report scores."""
from __future__ import annotations

import csv
import statistics as stats
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "results/bert_conll2003_standard_report_fixed600/BERT_kshot_standard_report_fixed600.csv"
MODELS_ROOT = ROOT / "models/bert_conll_kshot_fixed_final600_report"
LONG_CSV = ROOT / "reports/mikke_data/data/main_ablation_scores_long_fixed600_final.csv"
MEAN_CSV = ROOT / "reports/mikke_data/data/main_ablation_mean_scores_fixed600_final.csv"

BERT_NOTES = (
    "Standard k-shot BERT baseline; held-out test score. "
    "Fixed 600-step budget; final checkpoint; no early stopping or best-checkpoint selection."
)
POLICY_SUFFIX = (
    " Final fixed-600 checkpoint policy for BERT baselines and distilled students; "
    "no early stopping or best-checkpoint selection."
)


def load_bert_test_scores() -> dict[tuple[int, int], tuple[float, str]]:
    scores: dict[tuple[int, int], tuple[float, str]] = {}
    with RESULTS_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["Eval_Split"] != "test":
                continue
            key = (int(row["K_shot"]), int(row["Split_Seed"]))
            split_name = row["Split_Name"]
            summary = MODELS_ROOT / split_name / "training_summary.json"
            source = (
                summary.relative_to(ROOT).as_posix()
                if summary.exists()
                else RESULTS_CSV.relative_to(ROOT).as_posix()
            )
            scores[key] = (float(row["Span_Strict_F1"]), source)
    return scores


def patch_notes(notes: str) -> str:
    if "Final fixed-600" in notes or "fixed 600-step" in notes.lower():
        return notes
    return notes.rstrip(".") + "." + POLICY_SUFFIX


def make_individual_row(k_shot: int, seed: int, metric: float, source_path: str, today: str) -> dict[str, str]:
    return {
        "row_type": "individual",
        "plot_group": "main",
        "dataset": "conll2003",
        "task": "ner",
        "label_level": "coarse",
        "tagging_scheme": "BIO",
        "method_id": "bert",
        "method_label": "BERT",
        "model": "google-bert/bert-base-cased",
        "k_shot": str(k_shot),
        "split_seed": str(seed),
        "eval_split": "test",
        "metric_name": "span_strict_f1",
        "metric_label": "Strict span F1",
        "metric_value": f"{metric:.16f}".rstrip("0").rstrip("."),
        "metric_value_percent": f"{metric * 100:.16f}".rstrip("0").rstrip("."),
        "n_scores": "1",
        "expected_scores": "3",
        "completed_scores": "1",
        "score_source": "test",
        "status": "final_fixed600",
        "is_placeholder": "false",
        "needs_more_runs": "false",
        "source_path": source_path,
        "notes": BERT_NOTES,
        "last_updated": today,
    }


def make_mean_row(group: list[dict[str, str]], today: str) -> dict[str, str]:
    values = [float(row["metric_value"]) for row in group]
    mean_value = stats.mean(values)
    seeds = ";".join(row["split_seed"] for row in group)
    return {
        "row_type": "mean",
        "plot_group": "main",
        "dataset": "conll2003",
        "task": "ner",
        "label_level": "coarse",
        "tagging_scheme": "BIO",
        "method_id": "bert",
        "method_label": "BERT",
        "model": "google-bert/bert-base-cased",
        "k_shot": group[0]["k_shot"],
        "split_seed": seeds,
        "eval_split": "test",
        "metric_name": "span_strict_f1_mean",
        "metric_label": "Mean strict span F1",
        "metric_value": f"{mean_value:.16f}".rstrip("0").rstrip("."),
        "metric_value_percent": f"{mean_value * 100:.16f}".rstrip("0").rstrip("."),
        "n_scores": str(len(values)),
        "expected_scores": "3",
        "completed_scores": str(len(values)),
        "score_source": "test",
        "status": "final_fixed600",
        "is_placeholder": "false",
        "needs_more_runs": "true" if len(values) < 3 else "false",
        "source_path": ";".join(row["source_path"] for row in group),
        "notes": (
            f"Mean over split seeds {seeds}. Individual scores: "
            f"{';'.join(row['metric_value'] for row in group)}.{POLICY_SUFFIX}"
        ),
        "last_updated": today,
    }


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    bert_scores = load_bert_test_scores()
    today = date.today().isoformat()

    long_fields, long_rows = read_csv(LONG_CSV)
    non_bert = [row for row in long_rows if row["method_id"] != "bert"]
    bert_rows = []
    for (k_shot, seed), (metric, source) in sorted(bert_scores.items()):
        bert_rows.append(make_individual_row(k_shot, seed, metric, source, today))
    updated_long = non_bert + bert_rows
    write_csv(LONG_CSV, long_fields, updated_long)

    mean_fields, mean_rows = read_csv(MEAN_CSV)
    non_bert_mean = [row for row in mean_rows if row["method_id"] != "bert"]
    bert_mean = []
    for k_shot in ["5", "10", "20", "50"]:
        group = sorted(
            [row for row in bert_rows if row["k_shot"] == k_shot],
            key=lambda row: int(row["split_seed"]),
        )
        if group:
            bert_mean.append(make_mean_row(group, today))
    updated_mean = non_bert_mean + bert_mean
    write_csv(MEAN_CSV, mean_fields, updated_mean)

    print(f"Updated {len(bert_rows)} BERT individual rows in {LONG_CSV}")
    print(f"Updated {len(bert_mean)} BERT mean rows in {MEAN_CSV}")
    for k_shot in ["5", "10", "20", "50"]:
        group = [row for row in bert_rows if row["k_shot"] == k_shot]
        if not group:
            print(f"  WARNING: missing k={k_shot}")
            continue
        vals = [float(row["metric_value"]) for row in group]
        print(f"  k={k_shot}: n={len(vals)} mean test F1={stats.mean(vals)*100:.2f}")


if __name__ == "__main__":
    main()
