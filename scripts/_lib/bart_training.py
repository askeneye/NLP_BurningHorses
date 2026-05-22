from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from scripts._lib.repro_io import read_json_lines


class Seq2SeqJsonlDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def require_columns(rows: list[dict[str, Any]], required_columns: set[str], source_name: str) -> None:
    for row_index, row in enumerate(rows):
        missing_columns = required_columns.difference(row)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{source_name} row {row_index} missing required column(s): {missing}")


def make_seq2seq_collate_fn(tokenizer, max_source_length: int, max_target_length: int):
    def collate_fn(batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        input_texts = [row["input_text"] for row in batch]
        target_texts = [row["target_text"] for row in batch]
        tokenized = tokenizer(
            input_texts,
            max_length=max_source_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        target_tokenized = tokenizer(
            text_target=target_texts,
            max_length=max_target_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels = target_tokenized["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        tokenized["labels"] = labels
        return tokenized

    return collate_fn


def compute_eval_loss(model, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu()))
    if not losses:
        return 0.0
    return float(sum(losses) / len(losses))


@dataclass(frozen=True)
class BartTrainingJob:
    split_name: str
    pattern_id: str
    source_pattern_id: str
    model_name: str
    train_file: Path
    mini_val_file: Path
    output_dir: Path
    max_source_length: int
    max_target_length: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    max_steps: int
    eval_steps: int
    seed: int
    run_tag: str = "promoted"


def train_bart_job(job: BartTrainingJob) -> dict[str, Any]:
    torch.manual_seed(job.seed)
    started_at = time.time()

    train_rows = read_json_lines(job.train_file)
    mini_val_rows = read_json_lines(job.mini_val_file)
    require_columns(train_rows, {"input_text", "target_text"}, str(job.train_file))
    require_columns(mini_val_rows, {"input_text", "target_text"}, str(job.mini_val_file))

    tokenizer = AutoTokenizer.from_pretrained(job.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(job.model_name)
    tokenizer.model_max_length = job.max_source_length

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    collate_fn = make_seq2seq_collate_fn(tokenizer, job.max_source_length, job.max_target_length)
    train_loader = DataLoader(
        Seq2SeqJsonlDataset(train_rows),
        batch_size=job.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        Seq2SeqJsonlDataset(mini_val_rows),
        batch_size=job.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=job.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    job.output_dir.mkdir(parents=True, exist_ok=True)
    last_step_dir = job.output_dir / "last_step"
    train_history_path = job.output_dir / "train_history.jsonl"
    eval_history_path = job.output_dir / "eval_history.jsonl"
    train_history_path.unlink(missing_ok=True)
    eval_history_path.unlink(missing_ok=True)

    iterator = iter(train_loader)
    total_loss = 0.0
    global_step = 0
    best_eval_loss: float | None = None
    eval_history: list[dict[str, Any]] = []

    progress = tqdm(
        total=job.max_steps,
        desc=f"{job.pattern_id}:train",
        dynamic_ncols=True,
        mininterval=5.0,
        initial=global_step,
    )
    while global_step < job.max_steps:
        model.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(**batch)
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        global_step += 1
        loss_value = float(loss.detach().cpu())
        total_loss += loss_value
        progress.update(1)
        progress.set_postfix(
            loss=f"{loss_value:.4f}",
            avg=f"{(total_loss / global_step):.4f}",
        )
        train_record = {
            "step": global_step,
            "loss": loss_value,
            "avg_loss_so_far": total_loss / global_step,
        }
        with train_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(train_record, ensure_ascii=False) + "\n")

        if global_step % job.eval_steps != 0:
            continue
        eval_loss = compute_eval_loss(model, eval_loader, device)
        if best_eval_loss is None or eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
        eval_record = {
            "step": global_step,
            "mini_val_loss": eval_loss,
            "best_mini_val_loss": best_eval_loss,
        }
        eval_history.append(eval_record)
        progress.set_postfix(
            loss=f"{loss_value:.4f}",
            avg=f"{(total_loss / global_step):.4f}",
            val=f"{eval_loss:.4f}",
        )
        with eval_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(eval_record, ensure_ascii=False) + "\n")

    progress.close()
    last_step_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(last_step_dir)
    tokenizer.save_pretrained(last_step_dir)

    elapsed_seconds = time.time() - started_at
    summary = {
        "experiment": {
            "dataset": "conll2003",
            "method": "pet_oada",
            "model_family": "bart_base",
            "kshot_split": job.split_name,
            "pattern": job.pattern_id,
            "source_pattern_id": job.source_pattern_id,
            "seed": job.seed,
            "model_selection_strategy": "last_step",
            "use_early_stopping": False,
            "selection_policy": "last_step",
            "run_tag": job.run_tag,
        },
        "model_name": job.model_name,
        "train_file": str(job.train_file),
        "mini_val_file": str(job.mini_val_file),
        "output_dir": str(job.output_dir),
        "selected_model_dir": str(last_step_dir),
        "steps_trained": global_step,
        "avg_loss": total_loss / global_step if global_step else 0.0,
        "best_mini_val_loss": best_eval_loss,
        "train_rows": len(train_rows),
        "mini_val_rows": len(mini_val_rows),
        "total_training_seconds": elapsed_seconds,
        "train_history_file": str(train_history_path),
        "eval_history_file": str(eval_history_path),
        "eval_history": eval_history,
    }
    with (job.output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary
