from __future__ import annotations

import csv
from datetime import date
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(path for path in [SCRIPT_DIR, *SCRIPT_DIR.parents] if (path / "pyproject.toml").exists())
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_aug_train import (  # noqa: E402
    _iter_pattern_bank,
    apply_pet_wrapper,
    detokenize_tokens,
    extract_and_group_entities,
    generate_oada_target,
    get_oada_permutations,
    load_jsonl,
)

METHOD = "pet_oada_verbalized_diverse"
BART_SUFFIX = "xe_verbalized_diverse_final2000"
ENSEMBLE_ROOT = "data/interim/conll2003_ensemble_teachers/bart_base/pet_oada_verbalized_diverse_final2000"
FILTERED_ROOT = "data/interim/conll2003_ensemble_teachers/bart_base/pet_oada_verbalized_diverse_final2000_filtered"
SOFT_ROOT = "data/interim/conll2003_soft_labels/bart_base/pet_oada_verbalized_diverse"
TRAIN_ROOT = PROJECT_ROOT / "data/interim/conll2003_kshot_seq2seq/train/pet_oada_verbalized_diverse"
INFER_ROOT = PROJECT_ROOT / "data/interim/conll2003_kshot_seq2seq/inference/first_to_last_verbalized_diverse"
KSHOT_ROOT = PROJECT_ROOT / "data/interim/conll2003_kshot_bert"
BART_ROOT = PROJECT_ROOT / "models/conll2003_kshot_seq2seq/bart_base/pet_oada_verbalized_diverse"
BERT_ROOT = PROJECT_ROOT / "models/bert_conll_distilled"

LOW_K_SPLITS = ["k10_seed42", "k10_seed142", "k10_seed242"]
HIGH_K_SPLITS = [
    "k20_seed42",
    "k20_seed142",
    "k20_seed242",
    "k50_seed42",
    "k50_seed142",
    "k50_seed242",
]
ALL_SUITE_SPLITS = LOW_K_SPLITS + HIGH_K_SPLITS
K5_SPLITS = ["k5_seed42", "k5_seed142", "k5_seed242"]
LOW_K_PATTERNS = [f"pattern_{index:02d}" for index in range(1, 11)]
HIGH_K_PATTERNS = [f"pattern_{index:02d}" for index in range(1, 6)]

VERBALIZER_BANK = {
    "ORG": ["organization", "organisation", "company", "institution", "corporation", "agency", "group", "association", "body"],
    "PER": ["person", "individual", "human", "someone", "name", "identity", "figure", "subject", "character", "being"],
    "LOC": ["location", "place", "region", "area", "site", "territory", "venue", "country", "city", "destination"],
    "MISC": ["other", "concept", "thing", "term", "category", "label", "entity", "object", "event", "item"],
}
OFFSETS = {"ORG": 0, "PER": 1, "LOC": 2, "MISC": 3}
METHODS = {
    "pat_perm_verb_bart_ensemble_vote": {
        "label": "Pat+Perm+Verb BART ensemble",
        "model": "facebook/bart-base",
        "notes": "Direct hard-argmax vote from the Pat+Perm+Verb BART ensemble.",
    },
    "pat_perm_verb_bart_ensemble_to_bert": {
        "label": "Pat+Perm+Verb BART ensemble -> BERT",
        "model": "google-bert/bert-base-cased",
        "notes": "BERT distilled from top_1000 hard-argmax labels produced by the Pat+Perm+Verb BART ensemble.",
    },
}


def verbalizer_for_pattern(pattern_index: int) -> dict[str, str]:
    return {
        label: words[(pattern_index + OFFSETS[label]) % len(words)]
        for label, words in VERBALIZER_BANK.items()
    }


def patterns_for_split(split: str) -> list[str]:
    return LOW_K_PATTERNS if split.startswith("k10") or split.startswith("k5") else HIGH_K_PATTERNS


def run_python(script: Path, env: dict[str, str]) -> None:
    merged_env = os.environ.copy()
    merged_env.update(env)
    subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT, env=merged_env, check=True)


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_train_assets(splits: list[str]) -> None:
    pattern_bank = yaml.safe_load((SCRIPT_DIR / "patterns.yaml").read_text(encoding="utf-8"))
    patterns = _iter_pattern_bank(pattern_bank, "patterns")
    mapping = yaml.safe_load((SCRIPT_DIR.parent / "mapping.yaml").read_text(encoding="utf-8"))["conll2003"]
    schema = mapping["entity_types"]
    id_to_label = mapping["id_to_label"]

    for split in splits:
        examples = load_jsonl(KSHOT_ROOT / split / "train.jsonl")
        split_manifest: dict[str, dict[str, object]] = {}
        for pattern_index, pattern_id in enumerate(patterns_for_split(split)):
            source_pattern = patterns[pattern_index]
            verbalizer = verbalizer_for_pattern(pattern_index)
            output_path = TRAIN_ROOT / split / f"{pattern_id}.jsonl"
            if output_path.exists():
                split_manifest[pattern_id] = {
                    "source_pattern_id": source_pattern["id"],
                    "source_template": source_pattern["template"],
                    "label_verbalizer": verbalizer,
                    "output_file": output_path.relative_to(PROJECT_ROOT).as_posix(),
                    "skipped_existing": True,
                }
                continue

            rows = []
            for example in examples:
                tokens = example["tokens"]
                sentence_text = detokenize_tokens(tokens)
                entities_by_type = extract_and_group_entities(tokens, example["ner_tags"], id_to_label)
                for order in get_oada_permutations(schema):
                    rows.append(
                        {
                            "input_text": apply_pet_wrapper(sentence_text, order, source_pattern["template"], verbalizer),
                            "target_text": generate_oada_target(entities_by_type, order, verbalizer, label_separator=" "),
                        }
                    )
            write_jsonl(output_path, rows)
            split_manifest[pattern_id] = {
                "source_pattern_id": source_pattern["id"],
                "source_template": source_pattern["template"],
                "label_verbalizer": verbalizer,
                "source_rows": len(examples),
                "generated_rows": len(rows),
                "output_file": output_path.relative_to(PROJECT_ROOT).as_posix(),
            }
            print(f"Generated {output_path} rows={len(rows)}")

        manifest_path = TRAIN_ROOT / split / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_eval_assets() -> None:
    pattern_bank = yaml.safe_load((SCRIPT_DIR / "patterns.yaml").read_text(encoding="utf-8"))
    patterns = _iter_pattern_bank(pattern_bank, "patterns")
    id_to_label = {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-LOC",
        6: "I-LOC",
        7: "B-MISC",
        8: "I-MISC",
    }
    schema = ["PER", "ORG", "LOC", "MISC"]
    eval_files = {
        "mini_val": KSHOT_ROOT / "mini_val.jsonl",
        "validation": KSHOT_ROOT / "validation.jsonl",
        "test": KSHOT_ROOT / "test.jsonl",
    }

    for pattern_index, pattern_id in enumerate(LOW_K_PATTERNS):
        source_pattern = patterns[pattern_index]
        verbalizer = verbalizer_for_pattern(pattern_index)
        for eval_name, eval_path in eval_files.items():
            output_path = INFER_ROOT / eval_name / f"{pattern_id}.jsonl"
            if output_path.exists():
                continue
            rows = []
            for row in load_jsonl(eval_path):
                tokens = row["tokens"]
                sentence = detokenize_tokens(tokens)
                entities_by_type = extract_and_group_entities(tokens, row["ner_tags"], id_to_label)
                rows.append(
                    {
                        "input_text": source_pattern["template"].format(
                            SEN=sentence,
                            PERM="first to last",
                            sentence_text=sentence,
                            order="first to last",
                        ),
                        "target_text": generate_oada_target(entities_by_type, tuple(schema), verbalizer),
                    }
                )
            write_jsonl(output_path, rows)
            print(f"Generated {output_path} rows={len(rows)}")


def train_bart_members(splits: list[str]) -> None:
    pattern_bank = yaml.safe_load((SCRIPT_DIR / "patterns.yaml").read_text(encoding="utf-8"))
    source_patterns = _iter_pattern_bank(pattern_bank, "patterns")
    for split in splits:
        for pattern_index, pattern_id in enumerate(patterns_for_split(split)):
            output_dir = BART_ROOT / split / f"{pattern_id}_{BART_SUFFIX}"
            if (output_dir / "training_summary.json").exists():
                print(f"Skipping existing BART training {split} {pattern_id}")
                continue
            source_pattern_id = source_patterns[pattern_index]["id"]
            print(f"Training Pat+Perm+Verb BART {split} {pattern_id} source={source_pattern_id}")
            run_python(
                SCRIPT_DIR / "run_bart_oada_xe.py",
                {
                    "BART_METHOD": METHOD,
                    "BART_LOSS_TYPE": "xe",
                    "BART_SEED": "42",
                    "BART_KSHOT_SPLIT": split,
                    "BART_PATTERN": pattern_id,
                    "BART_SOURCE_PATTERN_ID": source_pattern_id,
                    "BART_TRAIN_FILE": f"data/interim/conll2003_kshot_seq2seq/train/pet_oada_verbalized_diverse/{split}/{pattern_id}.jsonl",
                    "BART_MINI_VAL_FILE": f"data/interim/conll2003_kshot_seq2seq/inference/first_to_last_verbalized_diverse/mini_val/{pattern_id}.jsonl",
                    "BART_MINI_VAL_GOLD_FILE": "data/interim/conll2003_kshot_bert/mini_val.jsonl",
                    "BART_VALIDATION_FILE": f"data/interim/conll2003_kshot_seq2seq/inference/first_to_last_verbalized_diverse/validation/{pattern_id}.jsonl",
                    "BART_VALIDATION_GOLD_FILE": "data/interim/conll2003_kshot_bert/validation.jsonl",
                    "BART_OUTPUT_DIR": str(output_dir),
                    "BART_MAX_STEPS": "2000",
                    "BART_EVAL_STEPS": "100",
                    "BART_USE_EARLY_STOPPING": "false",
                    "BART_MODEL_SELECTION_STRATEGY": "final_step",
                    "BART_OADA_TAU_START": "0.0",
                    "BART_OADA_TAU_END": "1.0",
                    "BART_OADA_TAU_WARMUP_STEPS": "2000",
                    "BART_OADA_CANDIDATE_CAP": "24",
                    "BART_OADA_CANDIDATE_SEED": "42",
                },
            )


def generate_hard_labels(splits: list[str]) -> None:
    for split in splits:
        for pattern_id in patterns_for_split(split):
            model_dir = BART_ROOT / split / f"{pattern_id}_{BART_SUFFIX}"
            for eval_split in ["unlabeled_pool", "test"]:
                output_dir = PROJECT_ROOT / SOFT_ROOT / split / f"{pattern_id}_{BART_SUFFIX}" / eval_split
                if (output_dir / "hard_predictions.jsonl").exists():
                    print(f"Skipping existing hard labels {split} {pattern_id} {eval_split}")
                    continue
                input_file = (
                    f"data/interim/conll2003_kshot_bert/{split}/unlabeled_pool.jsonl"
                    if eval_split == "unlabeled_pool"
                    else "data/interim/conll2003_kshot_bert/test.jsonl"
                )
                print(f"Generating hard labels {split} {pattern_id} {eval_split}")
                run_python(
                    SCRIPT_DIR / "generate_bart_member_soft_labels.py",
                    {
                        "SOFT_LABEL_DATASET": "conll2003",
                        "SOFT_LABEL_METHOD": METHOD,
                        "SOFT_LABEL_MODEL_FAMILY": "bart_base",
                        "SOFT_LABEL_KSHOT_SPLIT": split,
                        "SOFT_LABEL_PATTERN": pattern_id,
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


def aggregate_filter_and_distill(splits: list[str]) -> None:
    for split in splits:
        patterns = patterns_for_split(split)
        run_python(
            SCRIPT_DIR / "aggregate_bart_ensemble_votes.py",
            {
                "ENSEMBLE_TEACHER_SPLITS": split,
                "ENSEMBLE_TEACHER_PATTERNS": ",".join(patterns),
                "ENSEMBLE_TEACHER_EVAL_SPLITS": "unlabeled_pool,test",
                "ENSEMBLE_TEACHER_SOURCE_SUFFIX": BART_SUFFIX,
                "ENSEMBLE_TEACHER_INPUT_ROOT": SOFT_ROOT,
                "ENSEMBLE_TEACHER_OUTPUT_ROOT": ENSEMBLE_ROOT,
            },
        )
        run_python(
            SCRIPT_DIR / "filter_ensemble_teacher_subsets.py",
            {
                "FILTER_TEACHER_SPLITS": split,
                "FILTER_TEACHER_SIZES": "1000",
                "FILTER_TEACHER_INPUT_ROOT": ENSEMBLE_ROOT,
                "FILTER_TEACHER_OUTPUT_ROOT": FILTERED_ROOT,
            },
        )
        run_python(
            SCRIPT_DIR / "evaluate_ensemble_hard_predictions.py",
            {
                "ENSEMBLE_EVAL_INPUT_FILE": f"{ENSEMBLE_ROOT}/{split}/test/ensemble_hard_argmax.jsonl",
                "ENSEMBLE_EVAL_OUTPUT_FILE": f"reports/ensemble_tagging/{split}_pat_perm_verb_bart_ensemble_test_metrics.json",
            },
        )
        output_dir = BERT_ROOT / split / "pet_oada_verbalized_diverse_ensemble_filtered/top_1000_hard_argmax"
        if (output_dir / "training_summary.json").exists():
            print(f"Skipping existing BERT distillation {split}")
            continue
        run_python(
            PROJECT_ROOT / "nlp_burninghorses/bert_conll_distill.py",
            {
                "DISTILL_SPLIT": split,
                "DISTILL_TEACHER_SPLIT": "unlabeled_pool_top_1000",
                "DISTILL_TEACHER_VARIANT": "hard_argmax",
                "DISTILL_TRAIN_FILE": f"{FILTERED_ROOT}/{split}/unlabeled_pool/top_1000/ensemble_hard_argmax.jsonl",
                "DISTILL_OUTPUT_DIR": str(output_dir),
                "DISTILL_MAX_STEPS": "600",
                "DISTILL_EVAL_EVERY": "100",
                "DISTILL_PATIENCE": "5",
                "DISTILL_EARLY_STOPPING": "false",
                "DISTILL_USE_BEST_CHECKPOINT": "false",
                "DISTILL_SEED": "42",
            },
        )


def metric_from_direct(split: str) -> tuple[float, Path]:
    path = PROJECT_ROOT / f"reports/ensemble_tagging/{split}_pat_perm_verb_bart_ensemble_test_metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["metrics"]["Span_Strict_F1"]), path


def metric_from_distill(split: str) -> tuple[float, Path]:
    path = BERT_ROOT / split / "pet_oada_verbalized_diverse_ensemble_filtered/top_1000_hard_argmax/training_summary.json"
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return float(data["eval_results"]["test"]["Span_Strict_F1"]), path


def refresh_ablation_tables() -> None:
    data_dir = PROJECT_ROOT / "reports/mikke_data/data"
    individual_path = data_dir / "main_ablation_individual_scores.csv"
    mean_path = data_dir / "main_ablation_mean_scores.csv"
    long_path = data_dir / "main_ablation_scores_long.csv"
    today = date.today().isoformat()
    all_splits = K5_SPLITS + ALL_SUITE_SPLITS

    def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)

    def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def base_row(method_id: str, k_shot: str, split_seed: str, row_type: str) -> dict[str, str]:
        info = METHODS[method_id]
        return {
            "row_type": row_type,
            "plot_group": "main",
            "dataset": "conll2003",
            "task": "ner",
            "label_level": "coarse",
            "tagging_scheme": "BIO",
            "method_id": method_id,
            "method_label": info["label"],
            "model": info["model"],
            "k_shot": k_shot,
            "split_seed": split_seed,
            "eval_split": "test",
            "metric_name": "span_strict_f1" if row_type == "individual" else "span_strict_f1_mean",
            "metric_label": "Strict span F1" if row_type == "individual" else "Mean strict span F1",
            "metric_value": "",
            "metric_value_percent": "",
            "n_scores": "1" if row_type == "individual" else "0",
            "expected_scores": "3",
            "completed_scores": "1" if row_type == "individual" else "0",
            "score_source": "test",
            "status": "ready_for_draft",
            "is_placeholder": "false",
            "needs_more_runs": "false",
            "source_path": "",
            "notes": info["notes"],
            "last_updated": today,
        }

    def available_individual_rows(method_id: str) -> list[dict[str, str]]:
        getter = metric_from_direct if method_id.endswith("_vote") else metric_from_distill
        rows = []
        for split in all_splits:
            seed = split.rsplit("seed", 1)[1]
            k_shot = split.split("_", 1)[0].lstrip("k")
            try:
                metric, source_path = getter(split)
            except FileNotFoundError:
                continue
            row = base_row(method_id, k_shot, seed, "individual")
            row.update(
                {
                    "metric_value": repr(metric),
                    "metric_value_percent": repr(metric * 100),
                    "source_path": source_path.relative_to(PROJECT_ROOT).as_posix(),
                }
            )
            rows.append(row)
        return rows

    def mean_rows(method_id: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        output = []
        rows_by_k: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            rows_by_k.setdefault(row["k_shot"], []).append(row)
        for k_shot in ["5", "10", "20", "50"]:
            group = sorted(rows_by_k.get(k_shot, []), key=lambda row: int(row["split_seed"]))
            if not group:
                placeholder = base_row(method_id, k_shot, "42;142;242", "mean")
                placeholder.update(
                    {
                        "score_source": "pending",
                        "status": "pending",
                        "is_placeholder": "true",
                        "needs_more_runs": "true",
                        "notes": f"Placeholder for future Pat+Perm+Verb k={k_shot} runs.",
                    }
                )
                output.append(placeholder)
                continue
            values = [float(row["metric_value"]) for row in group]
            mean_value = sum(values) / len(values)
            row = base_row(method_id, k_shot, ";".join(row["split_seed"] for row in group), "mean")
            row.update(
                {
                    "metric_value": repr(mean_value),
                    "metric_value_percent": repr(mean_value * 100),
                    "n_scores": str(len(values)),
                    "completed_scores": str(len(values)),
                    "source_path": ";".join(row["source_path"] for row in group),
                    "needs_more_runs": "true" if len(values) < 3 else "false",
                    "notes": (
                        f"Mean over split seeds {row['split_seed']}. Individual scores: "
                        f"{';'.join(item['metric_value'] for item in group)}. {METHODS[method_id]['notes']}"
                    ),
                }
            )
            output.append(row)
        return output

    def sort_key(row: dict[str, str]) -> tuple[int, int, int, str]:
        method_order = [
            "bert",
            "pat_perm_bart",
            "pat_perm_bart_to_bert",
            "pat_bart_ensemble_to_bert",
            "pat_perm_bart_ensemble_vote",
            "pat_perm_bart_ensemble_to_bert",
            "pat_perm_verb_bart_ensemble_vote",
            "pat_perm_verb_bart_ensemble_to_bert",
            "seed_perm_bart_ensemble_vote",
            "seed_perm_bart_ensemble_to_bert",
        ]
        return (
            method_order.index(row["method_id"]) if row["method_id"] in method_order else 999,
            int(row["k_shot"]) if row.get("k_shot") else 999,
            int(str(row.get("split_seed", "999")).split(";")[0]) if row.get("split_seed") else 999,
            row.get("row_type", ""),
        )

    individual_fields, individual_rows = read_csv(individual_path)
    mean_fields, existing_mean_rows = read_csv(mean_path)
    long_fields, long_rows = read_csv(long_path)
    method_ids = set(METHODS)
    new_individual_rows = []
    new_mean_rows = []
    for method_id in METHODS:
        rows = available_individual_rows(method_id)
        new_individual_rows.extend(rows)
        new_mean_rows.extend(mean_rows(method_id, rows))
    individual_rows = [row for row in individual_rows if row["method_id"] not in method_ids] + new_individual_rows
    existing_mean_rows = [row for row in existing_mean_rows if row["method_id"] not in method_ids] + new_mean_rows
    long_rows = [row for row in long_rows if row["method_id"] not in method_ids] + new_individual_rows
    write_csv(individual_path, individual_fields, sorted(individual_rows, key=sort_key))
    write_csv(mean_path, mean_fields, sorted(existing_mean_rows, key=sort_key))
    write_csv(long_path, long_fields, sorted(long_rows, key=sort_key))
    run_python(PROJECT_ROOT / "reports/mikke_data/scripts/plot_ablation_kshot.py", {})


def main() -> None:
    splits = os.environ.get("PAT_PERM_VERB_SPLITS")
    suite_splits = [item.strip() for item in splits.split(",") if item.strip()] if splits else ALL_SUITE_SPLITS
    generate_train_assets(suite_splits)
    generate_eval_assets()
    train_bart_members(suite_splits)
    generate_hard_labels(suite_splits)
    aggregate_filter_and_distill(suite_splits)
    refresh_ablation_tables()


if __name__ == "__main__":
    main()
