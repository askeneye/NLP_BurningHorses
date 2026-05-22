from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from scripts._lib.repro_io import read_json_lines, write_json


@dataclass(frozen=True)
class BertDistillJob:
    split_name: str
    teacher_variant: str
    train_file: Path
    metadata_file: Path
    output_dir: Path
    model_name: str
    max_steps: int
    train_batch_size: int
    learning_rate: float
    max_length: int
    seed: int
    run_tag: str
    filter_top_k: int | None
    max_no_entity_fraction: float
    output_name_override: str | None = None


def resolve_step06_paths(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    split_name = str(config.get("default_profile", {}).get("split_name", "k5_seed242"))
    paths = config.get("paths", {})
    interim_root = root / str(paths.get("interim_root", "reproduction/data/interim"))
    models_root = root / str(paths.get("models_root", "reproduction/models"))
    logs_root = root / str(paths.get("logs_root", "reproduction/logs"))
    split_root = interim_root / split_name
    return {
        "split_name": split_name,
        "split_root": split_root,
        "labels_root": split_root / "ensemble_labels",
        "base_root": split_root / "base",
        "models_root": models_root,
        "logs_root": logs_root,
        "step06_manifest": logs_root / "step06_train_bert_manifest.json",
    }


def _teacher_variant_file(teacher_variant: str, temperature: float) -> str:
    if teacher_variant == "vote_normalized":
        return "soft_labels_normalized.jsonl"
    if teacher_variant == "vote_temp2":
        return f"soft_labels_temp{temperature:g}.jsonl"
    if teacher_variant == "hard_argmax":
        return "hard_labels.jsonl"
    raise ValueError("teacher_variant must be one of: vote_normalized, vote_temp2, hard_argmax")


def build_distill_job(
    *,
    config: dict[str, Any],
    resolved_paths: dict[str, Any],
    max_steps_override: int | None,
    train_file_override: Path | None = None,
    output_name_override: str | None = None,
    run_tag_override: str | None = None,
) -> BertDistillJob:
    student_cfg = config.get("student", {})
    teacher_outputs_cfg = config.get("teacher_outputs", {})
    teacher_variant = str(teacher_outputs_cfg.get("soft_label_variant", "vote_normalized"))
    temperature = float(teacher_outputs_cfg.get("temperature", 2.0))
    default_max_steps = int(student_cfg.get("max_steps", 600))
    max_steps = int(max_steps_override) if max_steps_override is not None else default_max_steps
    run_tag = run_tag_override or ("promoted" if max_steps == default_max_steps else f"smoke_s{max_steps}")
    filter_top_k_raw = teacher_outputs_cfg.get("filter_top_k")
    filter_top_k = None if train_file_override is not None else (int(filter_top_k_raw) if filter_top_k_raw is not None else None)
    max_no_entity_fraction = float(teacher_outputs_cfg.get("max_no_entity_fraction", 0.1))

    output_name = output_name_override or (teacher_variant if run_tag == "promoted" else f"{teacher_variant}__{run_tag}")
    output_dir = (
        Path(resolved_paths["models_root"]) / "student" / str(resolved_paths["split_name"]) / output_name
    )
    train_file = train_file_override or (
        Path(resolved_paths["labels_root"])
        / "unlabeled_pool"
        / _teacher_variant_file(teacher_variant, temperature)
    )
    metadata_file = Path(resolved_paths["base_root"]) / "metadata.json"

    return BertDistillJob(
        split_name=str(resolved_paths["split_name"]),
        teacher_variant=teacher_variant,
        train_file=train_file,
        metadata_file=metadata_file,
        output_dir=output_dir,
        model_name=str(student_cfg.get("model_name", "google-bert/bert-base-cased")),
        max_steps=max_steps,
        train_batch_size=int(student_cfg.get("train_batch_size", 32)),
        learning_rate=float(student_cfg.get("learning_rate", 5e-5)),
        max_length=128,
        seed=42,
        run_tag=run_tag,
        filter_top_k=filter_top_k,
        max_no_entity_fraction=max_no_entity_fraction,
        output_name_override=output_name_override,
    )


def _row_agreement_score(row: dict[str, Any]) -> dict[str, Any]:
    predicted_tags = row["predicted_tags"]
    vote_confidences = [float(value) for value in row.get("vote_confidences", [])]
    if not vote_confidences:
        vote_confidences = [1.0] * len(predicted_tags)
    entity_indices = [index for index, label in enumerate(predicted_tags) if label != "O"]
    has_entity = bool(entity_indices)
    if entity_indices:
        entity_confidence = sum(vote_confidences[index] for index in entity_indices) / len(entity_indices)
        min_entity_confidence = min(vote_confidences[index] for index in entity_indices)
    else:
        entity_confidence = 0.0
        min_entity_confidence = 0.0
    mean_confidence = sum(vote_confidences) / len(vote_confidences) if vote_confidences else 0.0
    unanimous_fraction = (
        sum(1 for confidence in vote_confidences if confidence == 1.0) / len(vote_confidences)
        if vote_confidences
        else 0.0
    )
    return {
        "has_entity": has_entity,
        "entity_token_count": len(entity_indices),
        "entity_confidence": entity_confidence,
        "min_entity_confidence": min_entity_confidence,
        "mean_confidence": mean_confidence,
        "unanimous_fraction": unanimous_fraction,
    }


def _select_top_k_indices(
    hard_rows: list[dict[str, Any]], top_k: int, max_no_entity_fraction: float
) -> tuple[list[int], dict[str, Any]]:
    scored_rows = [(row_index, _row_agreement_score(row)) for row_index, row in enumerate(hard_rows)]
    entity_rows = [item for item in scored_rows if item[1]["has_entity"]]
    no_entity_rows = [item for item in scored_rows if not item[1]["has_entity"]]
    ranked_entity_rows = sorted(
        entity_rows,
        key=lambda item: (
            item[1]["min_entity_confidence"],
            item[1]["entity_confidence"],
            item[1]["mean_confidence"],
            item[1]["entity_token_count"],
        ),
        reverse=True,
    )
    ranked_no_entity_rows = sorted(
        no_entity_rows,
        key=lambda item: (item[1]["mean_confidence"], item[1]["unanimous_fraction"]),
        reverse=True,
    )

    max_no_entity = max(0, int(top_k * max_no_entity_fraction))
    selected = ranked_entity_rows[:top_k]
    if len(selected) < top_k:
        selected.extend(ranked_no_entity_rows[: top_k - len(selected)])
    elif max_no_entity:
        selected_entity = selected[: top_k - max_no_entity]
        selected = selected_entity + ranked_no_entity_rows[:max_no_entity]

    indices = sorted(row_index for row_index, _ in selected[:top_k])
    selected_scores = [score for _, score in selected[:top_k]]
    stats = {
        "requested_subset_size": top_k,
        "selected_rows": len(indices),
        "selected_with_entities": sum(1 for score in selected_scores if score["has_entity"]),
        "selected_without_entities": sum(1 for score in selected_scores if not score["has_entity"]),
        "mean_entity_confidence": (
            sum(score["entity_confidence"] for score in selected_scores if score["has_entity"])
            / max(1, sum(1 for score in selected_scores if score["has_entity"]))
        ),
        "mean_token_confidence": sum(score["mean_confidence"] for score in selected_scores) / max(1, len(selected_scores)),
    }
    return indices, stats


def _prepare_filtered_train_file(job: BertDistillJob) -> tuple[Path, dict[str, Any] | None]:
    if job.filter_top_k is None:
        return job.train_file, None
    if job.teacher_variant != "hard_argmax":
        return job.train_file, None

    hard_rows = read_json_lines(job.train_file)
    indices, stats = _select_top_k_indices(hard_rows, top_k=job.filter_top_k, max_no_entity_fraction=job.max_no_entity_fraction)
    filtered_rows = [hard_rows[index] for index in indices]
    subset_dir = job.train_file.parent / f"top_{job.filter_top_k}"
    filtered_file = subset_dir / "hard_labels.jsonl"
    subset_dir.mkdir(parents=True, exist_ok=True)
    with filtered_file.open("w", encoding="utf-8") as handle:
        for row in filtered_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    filter_manifest = {
        "source_file": str(job.train_file),
        "filtered_file": str(filtered_file),
        "selection_policy": (
            "Rank entity-containing examples first by min entity-token vote confidence, "
            "then mean entity confidence and mean token confidence. Reserve up to configured "
            "fraction for high-confidence no-entity rows."
        ),
        "max_no_entity_fraction": job.max_no_entity_fraction,
        **stats,
    }
    write_json(subset_dir / "manifest.json", filter_manifest)
    return filtered_file, filter_manifest


class DistillDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,
        label_to_id: dict[str, int],
        teacher_variant: str,
        max_length: int,
    ) -> None:
        self.features: list[dict[str, torch.Tensor]] = []
        hard_variant = teacher_variant == "hard_argmax"
        num_labels = len(label_to_id)

        for row in rows:
            tokenized = tokenizer(
                row["tokens"],
                max_length=max_length,
                truncation=True,
                is_split_into_words=True,
            )
            word_ids = tokenized.word_ids()
            previous_word_id = None
            aligned_soft: list[list[float]] = []
            aligned_hard: list[int] = []
            loss_mask: list[int] = []

            for word_id in word_ids:
                if word_id is None or word_id == previous_word_id:
                    aligned_soft.append([0.0] * num_labels)
                    aligned_hard.append(-100)
                    loss_mask.append(0)
                else:
                    if hard_variant:
                        label_id = label_to_id[row["predicted_tags"][word_id]]
                        one_hot = [0.0] * num_labels
                        one_hot[label_id] = 1.0
                        aligned_soft.append(one_hot)
                        aligned_hard.append(label_id)
                    else:
                        distribution = [float(value) for value in row["soft_labels"][word_id]]
                        aligned_soft.append(distribution)
                        aligned_hard.append(max(range(len(distribution)), key=distribution.__getitem__))
                    loss_mask.append(1)
                previous_word_id = word_id

            feature = {key: torch.tensor(value, dtype=torch.long) for key, value in tokenized.items()}
            feature["soft_labels"] = torch.tensor(aligned_soft, dtype=torch.float32)
            feature["hard_labels"] = torch.tensor(aligned_hard, dtype=torch.long)
            feature["loss_mask"] = torch.tensor(loss_mask, dtype=torch.bool)
            self.features.append(feature)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.features[index]


class DistillCollator:
    def __init__(self, tokenizer) -> None:
        self.token_collator = DataCollatorForTokenClassification(tokenizer)

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        token_features = [
            {key: value for key, value in feature.items() if key not in {"soft_labels", "hard_labels", "loss_mask"}}
            for feature in features
        ]
        batch = self.token_collator(token_features)
        max_len = batch["input_ids"].shape[1]
        num_labels = features[0]["soft_labels"].shape[1]

        soft = torch.zeros((len(features), max_len, num_labels), dtype=torch.float32)
        hard = torch.full((len(features), max_len), -100, dtype=torch.long)
        mask = torch.zeros((len(features), max_len), dtype=torch.bool)
        for index, feature in enumerate(features):
            length = feature["soft_labels"].shape[0]
            soft[index, :length] = feature["soft_labels"]
            hard[index, :length] = feature["hard_labels"]
            mask[index, :length] = feature["loss_mask"]
        batch["soft_labels"] = soft
        batch["hard_labels"] = hard
        batch["loss_mask"] = mask
        return batch


def _distill_loss(logits: torch.Tensor, batch: dict[str, torch.Tensor], teacher_variant: str) -> torch.Tensor:
    if teacher_variant == "hard_argmax":
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), batch["hard_labels"].view(-1), ignore_index=-100)
    log_probs = F.log_softmax(logits, dim=-1)
    token_losses = -(batch["soft_labels"] * log_probs).sum(dim=-1)
    return token_losses[batch["loss_mask"]].mean()


def run_distillation(job: BertDistillJob, dry_run: bool) -> dict[str, Any]:
    print(
        f"distill_job split={job.split_name} teacher={job.teacher_variant} "
        f"steps={job.max_steps} output={job.output_dir}"
    )
    if dry_run:
        return {"status": "dry_run", "output_dir": str(job.output_dir)}

    if not job.train_file.exists():
        raise FileNotFoundError(f"Missing Step-05 teacher labels: {job.train_file}")
    if not job.metadata_file.exists():
        raise FileNotFoundError(f"Missing k-shot metadata: {job.metadata_file}")

    train_file, filter_manifest = _prepare_filtered_train_file(job)

    torch.manual_seed(job.seed)
    started_at = time.time()
    metadata = json.loads(job.metadata_file.read_text(encoding="utf-8"))
    label_list = [str(label) for label in metadata["label_list"]]
    label_to_id = {label: index for index, label in enumerate(label_list)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    teacher_rows = read_json_lines(train_file)

    tokenizer = AutoTokenizer.from_pretrained(job.model_name, use_fast=True)
    model_config = AutoConfig.from_pretrained(
        job.model_name,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    model = AutoModelForTokenClassification.from_pretrained(job.model_name, config=model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = DistillDataset(
        teacher_rows,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        teacher_variant=job.teacher_variant,
        max_length=job.max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=job.train_batch_size,
        shuffle=True,
        collate_fn=DistillCollator(tokenizer),
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=job.learning_rate)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    train_history_path = job.output_dir / "train_history.jsonl"
    train_history_path.unlink(missing_ok=True)

    total_loss = 0.0
    global_step = 0
    iterator = iter(dataloader)
    progress = tqdm(total=job.max_steps, desc="bert_student:train", dynamic_ncols=True, mininterval=5.0)
    while global_step < job.max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        model_inputs = {key: value for key, value in batch.items() if key not in {"soft_labels", "hard_labels", "loss_mask"}}

        model.train()
        optimizer.zero_grad()
        outputs = model(**model_inputs)
        loss = _distill_loss(outputs.logits, batch, job.teacher_variant)
        loss.backward()
        optimizer.step()

        global_step += 1
        loss_value = float(loss.detach().cpu())
        total_loss += loss_value
        progress.update(1)
        progress.set_postfix(
            loss=f"{loss_value:.4f}",
            avg=f"{(total_loss / global_step):.4f}",
        )
        with train_history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "step": global_step,
                        "loss": loss_value,
                        "avg_loss_so_far": total_loss / global_step,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    progress.close()

    last_step_dir = job.output_dir / "last_step"
    last_step_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(last_step_dir)
    tokenizer.save_pretrained(last_step_dir)

    summary = {
        "experiment": {
            "dataset": "conll2003",
            "split": job.split_name,
            "teacher_variant": job.teacher_variant,
            "selection_policy": "last_step",
            "run_tag": job.run_tag,
            "seed": job.seed,
        },
        "train_file": str(train_file),
        "output_dir": str(job.output_dir),
        "selected_model_dir": str(last_step_dir),
        "steps_trained": job.max_steps,
        "avg_loss": total_loss / job.max_steps if job.max_steps else 0.0,
        "train_rows": len(teacher_rows),
        "total_training_seconds": time.time() - started_at,
        "train_history_file": str(train_history_path),
    }
    if filter_manifest is not None:
        summary["filtering"] = filter_manifest
    write_json(job.output_dir / "training_summary.json", summary)
    return summary


def write_step06_run_manifest(
    *,
    manifest_path: Path,
    job: BertDistillJob,
    summary: dict[str, Any],
    dry_run: bool,
) -> Path:
    payload = {
        "dry_run": dry_run,
        "job": {
            "split_name": job.split_name,
            "teacher_variant": job.teacher_variant,
            "run_tag": job.run_tag,
            "train_file": str(job.train_file),
            "output_dir": str(job.output_dir),
            "max_steps": job.max_steps,
            "output_name_override": job.output_name_override,
        },
        "summary": summary,
    }
    write_json(manifest_path, payload)
    return manifest_path


def print_step06_summary(summary: dict[str, Any], manifest_path: Path, dry_run: bool) -> None:
    print("Prepared BERT student distillation")
    print(f"dry_run={dry_run}")
    if not dry_run:
        print(
            f"steps={summary['steps_trained']} train_rows={summary['train_rows']} "
            f"avg_loss={summary['avg_loss']:.4f}"
        )
    print(f"run_manifest={manifest_path}")
