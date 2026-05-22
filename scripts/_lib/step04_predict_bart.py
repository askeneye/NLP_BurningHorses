from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from scripts._lib.repro_io import load_yaml, read_json_lines, write_json
from scripts._lib.teacher_data import apply_pattern, detokenize_tokens, normalize_pattern_bank

BRACKETED_ENTITY_RE = re.compile(r"\[([^\]]+)\]\s*([A-Za-z][A-Za-z0-9_/-]*)")
ENTITY_TYPE_ALIASES = {
    "person": "PER",
    "individual": "PER",
    "human": "PER",
    "location": "LOC",
    "place": "LOC",
    "country": "LOC",
    "city": "LOC",
    "organization": "ORG",
    "organisation": "ORG",
    "company": "ORG",
    "institution": "ORG",
    "other": "MISC",
    "misc": "MISC",
    "miscellaneous": "MISC",
}


@dataclass(frozen=True)
class BartPredictionJob:
    split_name: str
    pattern_id: str
    source_pattern_id: str
    source_template: str
    model_dir: Path
    checkpoint_dir: Path
    output_root: Path
    max_steps: int
    run_tag: str


def _relative(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(resolved_root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def resolve_step04_paths(root: Path, config: dict[str, Any]) -> dict[str, Path | str]:
    split_name = str(config.get("default_profile", {}).get("split_name", "k5_seed242"))
    paths = config.get("paths", {})
    interim_root = root / str(paths.get("interim_root", "reproduction/data/interim"))
    models_root = root / str(paths.get("models_root", "reproduction/models"))
    logs_root = root / str(paths.get("logs_root", "reproduction/logs"))
    pattern_file = root / str(paths.get("pattern_bank_file", "scripts/assets/patterns.yaml"))
    return {
        "split_name": split_name,
        "interim_root": interim_root,
        "models_root": models_root,
        "logs_root": logs_root,
        "pattern_file": pattern_file,
        "split_root": interim_root / split_name,
        "kshot_root": interim_root / split_name / "base",
        "step03_manifest": logs_root / "step03_train_bart_manifest.json",
        "step04_manifest": logs_root / "step04_predict_bart_manifest.json",
    }


def parse_eval_splits(eval_splits_csv: str | None) -> list[str]:
    if not eval_splits_csv:
        return ["mini_val", "validation", "test", "unlabeled_pool"]
    parsed = [part.strip() for part in eval_splits_csv.split(",") if part.strip()]
    if not parsed:
        raise ValueError("Expected at least one split in --eval-splits.")
    return parsed


def load_step04_inputs(
    *,
    step03_manifest_path: Path,
    kshot_root: Path,
    split_name: str,
    pattern_file: Path,
    eval_splits: list[str],
) -> dict[str, Any]:
    if not step03_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing Step-03 run manifest at {step03_manifest_path}. "
            "Run step 03 first (or rerun it) before step 04."
        )

    step03_manifest = json.loads(step03_manifest_path.read_text(encoding="utf-8"))
    pattern_bank = normalize_pattern_bank(load_yaml(pattern_file), "patterns")
    patterns_by_source_id = {pattern["id"]: pattern for pattern in pattern_bank}

    metadata = json.loads((kshot_root / "metadata.json").read_text(encoding="utf-8"))
    label_list = [str(label) for label in metadata["label_list"]]
    id_to_label = {int(index): str(label) for index, label in metadata["id_to_label"].items()}
    entity_types = {label[2:] for label in label_list if label.startswith("B-")}

    split_rows: dict[str, list[dict[str, Any]]] = {}
    supported = {"mini_val", "validation", "test", "unlabeled_pool"}
    unknown = sorted(set(eval_splits).difference(supported))
    if unknown:
        raise ValueError(f"Unsupported eval split(s): {', '.join(unknown)}")
    for eval_split in eval_splits:
        if eval_split == "unlabeled_pool":
            split_file = kshot_root / "unlabeled_pool.jsonl"
        else:
            split_file = kshot_root / f"{eval_split}.jsonl"
        split_rows[eval_split] = read_json_lines(split_file)

    return {
        "step03_manifest": step03_manifest,
        "patterns_by_source_id": patterns_by_source_id,
        "split_rows": split_rows,
        "label_list": label_list,
        "id_to_label": id_to_label,
        "entity_types": entity_types,
    }


def build_prediction_jobs(
    *,
    root: Path,
    config: dict[str, Any],
    resolved_paths: dict[str, Path | str],
    inputs: dict[str, Any],
    pattern_limit: int | None,
    checkpoint_subdir_override: str | None,
) -> list[BartPredictionJob]:
    split_name = str(resolved_paths["split_name"])
    interim_root = Path(resolved_paths["interim_root"])
    teacher_cfg = config.get("teacher_ensemble", {})
    preferred_pattern_ids = list(teacher_cfg.get("pattern_ids", []))
    selection_strategy = str(teacher_cfg.get("train", {}).get("model_selection_strategy", "last_step"))
    checkpoint_subdir = checkpoint_subdir_override or (
        "last_step" if selection_strategy == "last_step" else "best_span_f1"
    )

    jobs: list[BartPredictionJob] = []
    all_jobs = list(inputs["step03_manifest"].get("jobs", []))
    if preferred_pattern_ids:
        all_jobs = [job for job in all_jobs if str(job.get("pattern_id")) in preferred_pattern_ids]
    if pattern_limit is not None:
        all_jobs = all_jobs[:pattern_limit]

    for job in all_jobs:
        pattern_id = str(job["pattern_id"])
        source_pattern_id = str(job["source_pattern_id"])
        max_steps = int(job["max_steps"])
        run_tag = str(job.get("run_tag", "promoted"))
        model_dir = Path(job["output_dir"])
        checkpoint_dir = model_dir / checkpoint_subdir
        pattern_record = inputs["patterns_by_source_id"].get(source_pattern_id)
        if pattern_record is None:
            raise ValueError(f"Missing source pattern in pattern bank: {source_pattern_id}")
        output_pattern_dir = f"{pattern_id}__{run_tag}" if run_tag != "promoted" else pattern_id
        output_root = interim_root / split_name / "teacher_predictions" / output_pattern_dir
        jobs.append(
            BartPredictionJob(
                split_name=split_name,
                pattern_id=pattern_id,
                source_pattern_id=source_pattern_id,
                source_template=str(pattern_record["template"]),
                model_dir=model_dir,
                checkpoint_dir=checkpoint_dir,
                output_root=output_root,
                max_steps=max_steps,
                run_tag=run_tag,
            )
        )

    return jobs


def normalize_entity_type(entity_type: str) -> str:
    cleaned = entity_type.strip()
    return ENTITY_TYPE_ALIASES.get(cleaned, ENTITY_TYPE_ALIASES.get(cleaned.lower(), cleaned.upper()))


def find_entity_span(tokens: list[str], entity_text: str, occupied_indices: set[int]) -> tuple[int, int] | None:
    normalized = entity_text.strip()
    if not normalized:
        return None
    for start_index in range(len(tokens)):
        for end_index in range(start_index + 1, len(tokens) + 1):
            indices = set(range(start_index, end_index))
            if indices.intersection(occupied_indices):
                continue
            if detokenize_tokens(tokens[start_index:end_index]) == normalized:
                return start_index, end_index
    return None


def generated_text_to_bio_tags(generated_text: str, tokens: list[str], entity_types: set[str]) -> list[str]:
    tags = ["O"] * len(tokens)
    occupied_indices: set[int] = set()
    for match in BRACKETED_ENTITY_RE.finditer(generated_text):
        entity_text = match.group(1).strip()
        entity_type = normalize_entity_type(match.group(2))
        if entity_type not in entity_types:
            continue
        span = find_entity_span(tokens, entity_text, occupied_indices)
        if span is None:
            continue
        start_index, end_index = span
        tags[start_index] = f"B-{entity_type}"
        for token_index in range(start_index + 1, end_index):
            tags[token_index] = f"I-{entity_type}"
        occupied_indices.update(range(start_index, end_index))
    return tags


def _gold_tags_from_row(row: dict[str, Any], id_to_label: dict[int, str]) -> list[str] | None:
    if "ner_tags" not in row:
        return None
    gold_tags: list[str] = []
    for tag in row["ner_tags"]:
        if isinstance(tag, int):
            gold_tags.append(id_to_label[tag])
        elif isinstance(tag, str) and tag.isdigit():
            gold_tags.append(id_to_label[int(tag)])
        else:
            gold_tags.append(str(tag))
    return gold_tags


def _run_member_predictions(
    *,
    job: BartPredictionJob,
    split_rows: dict[str, list[dict[str, Any]]],
    entity_types: set[str],
    id_to_label: dict[int, str],
    batch_size: int,
    max_source_length: int,
    max_generation_length: int,
    overwrite: bool,
) -> dict[str, Any]:
    if not job.checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {job.checkpoint_dir}")

    tokenizer = AutoTokenizer.from_pretrained(job.checkpoint_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(job.checkpoint_dir)
    tokenizer.model_max_length = max_source_length
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    job_summary: dict[str, Any] = {
        "pattern_id": job.pattern_id,
        "source_pattern_id": job.source_pattern_id,
        "model_dir": str(job.model_dir),
        "checkpoint_dir": str(job.checkpoint_dir),
        "output_root": str(job.output_root),
        "splits": {},
    }

    for split_name, rows in split_rows.items():
        split_dir = job.output_root / split_name
        hard_file = split_dir / "hard_predictions.jsonl"
        manifest_file = split_dir / "manifest.json"
        if not overwrite and hard_file.exists() and manifest_file.exists():
            job_summary["splits"][split_name] = {
                "status": "skipped_existing",
                "rows_written": len(read_json_lines(hard_file)),
                "output_file": str(hard_file),
            }
            continue

        started_at = time.time()
        hard_rows: list[dict[str, Any]] = []
        for start in tqdm(
            range(0, len(rows), batch_size),
            desc=f"{job.pattern_id}:{split_name}",
            dynamic_ncols=True,
            mininterval=5.0,
        ):
            batch = rows[start : start + batch_size]
            input_texts = [
                apply_pattern(
                    sentence_text=detokenize_tokens(row["tokens"]),
                    order_text="first to last",
                    pattern_template=job.source_template,
                )
                for row in batch
            ]
            tokenized = tokenizer(
                input_texts,
                max_length=max_source_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    max_length=max_generation_length,
                )
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for row_index, row in enumerate(batch):
                row_id = row.get("id", f"row-{start + row_index:06d}")
                generated_text = generated_texts[row_index].strip()
                hard_row: dict[str, Any] = {
                    "id": row_id,
                    "tokens": row["tokens"],
                    "predicted_tags": generated_text_to_bio_tags(generated_text, row["tokens"], entity_types),
                    "generated_text": generated_text,
                }
                if "ner_tags" in row:
                    hard_row["ner_tags"] = row["ner_tags"]
                    hard_row["gold_tags"] = _gold_tags_from_row(row, id_to_label)
                hard_rows.append(hard_row)

        split_dir.mkdir(parents=True, exist_ok=True)
        with hard_file.open("w", encoding="utf-8") as output_file:
            for hard_row in hard_rows:
                output_file.write(json.dumps(hard_row, ensure_ascii=False) + "\n")
        split_manifest = {
            "pattern_id": job.pattern_id,
            "run_tag": job.run_tag,
            "source_pattern_id": job.source_pattern_id,
            "source_template": job.source_template,
            "inference_order": "first to last",
            "input_rows": len(rows),
            "rows_written": len(hard_rows),
            "hard_prediction_file": str(hard_file),
            "elapsed_seconds": time.time() - started_at,
        }
        write_json(manifest_file, split_manifest)
        job_summary["splits"][split_name] = {
            "status": "written",
            "rows_written": len(hard_rows),
            "output_file": str(hard_file),
        }

    return job_summary


def run_prediction_jobs(
    *,
    jobs: list[BartPredictionJob],
    split_rows: dict[str, list[dict[str, Any]]],
    entity_types: set[str],
    id_to_label: dict[int, str],
    batch_size: int,
    max_source_length: int,
    max_generation_length: int,
    overwrite: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for job in jobs:
        print(
            f"predict_job split={job.split_name} pattern={job.pattern_id} "
            f"checkpoint={job.checkpoint_dir} output={job.output_root}"
        )
        if dry_run:
            summaries.append(
                {
                    "pattern_id": job.pattern_id,
                    "status": "dry_run",
                    "checkpoint_dir": str(job.checkpoint_dir),
                    "output_root": str(job.output_root),
                }
            )
            continue
        summaries.append(
            _run_member_predictions(
                job=job,
                split_rows=split_rows,
                entity_types=entity_types,
                id_to_label=id_to_label,
                batch_size=batch_size,
                max_source_length=max_source_length,
                max_generation_length=max_generation_length,
                overwrite=overwrite,
            )
        )
    return summaries


def write_step04_run_manifest(
    *,
    root: Path,
    manifest_path: Path,
    jobs: list[BartPredictionJob],
    summaries: list[dict[str, Any]],
    dry_run: bool,
) -> Path:
    payload = {
        "dry_run": dry_run,
        "jobs": [
            {
                "split_name": job.split_name,
                "pattern_id": job.pattern_id,
                "source_pattern_id": job.source_pattern_id,
                "model_dir": _relative(root, job.model_dir),
                "checkpoint_dir": _relative(root, job.checkpoint_dir),
                "output_root": _relative(root, job.output_root),
                "max_steps": job.max_steps,
                "run_tag": job.run_tag,
            }
            for job in jobs
        ],
        "summaries": summaries,
    }
    write_json(manifest_path, payload)
    return manifest_path


def print_step04_summary(
    jobs: list[BartPredictionJob],
    summaries: list[dict[str, Any]],
    manifest_path: Path,
    dry_run: bool,
) -> None:
    print("Prepared BART teacher predictions")
    print(f"job_count={len(jobs)} dry_run={dry_run}")
    if not dry_run:
        for summary in summaries:
            split_records = summary.get("splits", {})
            counts = ", ".join(f"{name}:{record['rows_written']}" for name, record in split_records.items())
            print(f"finished pattern={summary.get('pattern_id')} splits={counts}")
    print(f"run_manifest={manifest_path}")
