from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    set_seed,
)
from transformers.utils import logging as hf_logging

from bert_eval import evaluate_model, make_dataloader, tokenize_and_align_labels_fn


MODEL_NAME = "google-bert/bert-base-cased"
LEARNING_RATE = 2e-5
MAX_TRAIN_STEPS = 1000
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
EVAL_EVERY = 100
PATIENCE = 5
MIN_STEPS_BEFORE_EVAL = 100
MAX_LENGTH = 128


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = repo_root()
DATA_ROOT = PROJECT_ROOT / "data" / "interim" / "conll2003_kshot_bert"
TEACHER_ROOT = PROJECT_ROOT / "data" / "interim" / "conll2003_ensemble_teachers" / "bart_base" / "pet_oada"
MODELS_ROOT = PROJECT_ROOT / "models" / "bert_conll_distilled"


@dataclass(frozen=True)
class DistillConfig:
    split: str
    teacher_split: str
    teacher_variant: str
    train_file: Path
    mini_eval_file: Path
    validation_file: Path
    test_file: Path
    metadata_file: Path
    output_dir: Path
    max_steps: int
    eval_every: int
    patience: int
    min_steps_before_eval: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    seed: int
    max_length: int


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped_line = line.strip()
            if stripped_line:
                rows.append(json.loads(stripped_line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)


def dataset_from_jsonl(path: Path) -> Dataset:
    return Dataset.from_list(read_jsonl(path))


def teacher_file_for(split: str, teacher_split: str, variant: str) -> Path:
    filenames = {
        "vote_normalized": "ensemble_vote_normalized.jsonl",
        "vote_temp2": "ensemble_vote_temp2.jsonl",
        "hard_argmax": "ensemble_hard_argmax.jsonl",
        "verbalized_true_soft": "ensemble_verbalized_true_soft.jsonl",
        "verbalized_hard_argmax": "ensemble_verbalized_hard_argmax.jsonl",
    }
    if variant not in filenames:
        raise ValueError("teacher_variant must be one of: " + ", ".join(sorted(filenames)))
    return TEACHER_ROOT / split / teacher_split / filenames[variant]


class DistillationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,
        label_to_id: dict[str, int],
        *,
        teacher_variant: str,
        max_length: int,
    ) -> None:
        self.features: list[dict[str, torch.Tensor]] = []
        self.teacher_variant = teacher_variant

        for row in rows:
            tokenized = tokenizer(
                row["tokens"],
                max_length=max_length,
                truncation=True,
                is_split_into_words=True,
            )
            word_ids = tokenized.word_ids()
            previous_word_id = None
            aligned_soft_labels: list[list[float]] = []
            aligned_hard_labels: list[int] = []
            loss_mask: list[int] = []

            for word_id in word_ids:
                if word_id is None or word_id == previous_word_id:
                    aligned_soft_labels.append([0.0] * len(label_to_id))
                    aligned_hard_labels.append(-100)
                    loss_mask.append(0)
                else:
                    if teacher_variant in {"hard_argmax", "verbalized_hard_argmax"}:
                        label_id = label_to_id[row["predicted_tags"][word_id]]
                        one_hot = [0.0] * len(label_to_id)
                        one_hot[label_id] = 1.0
                        aligned_soft_labels.append(one_hot)
                        aligned_hard_labels.append(label_id)
                    else:
                        distribution = [float(value) for value in row["soft_labels"][word_id]]
                        aligned_soft_labels.append(distribution)
                        aligned_hard_labels.append(max(range(len(distribution)), key=distribution.__getitem__))
                    loss_mask.append(1)
                previous_word_id = word_id

            feature = {key: torch.tensor(value, dtype=torch.long) for key, value in tokenized.items()}
            feature["soft_labels"] = torch.tensor(aligned_soft_labels, dtype=torch.float32)
            feature["hard_labels"] = torch.tensor(aligned_hard_labels, dtype=torch.long)
            feature["loss_mask"] = torch.tensor(loss_mask, dtype=torch.bool)
            self.features.append(feature)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.features[index]


class DistillationCollator:
    def __init__(self, tokenizer) -> None:
        self.token_collator = DataCollatorForTokenClassification(tokenizer)

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        token_features = [
            {
                key: value
                for key, value in feature.items()
                if key not in {"soft_labels", "hard_labels", "loss_mask"}
            }
            for feature in features
        ]
        batch = self.token_collator(token_features)
        max_length = batch["input_ids"].shape[1]
        num_labels = features[0]["soft_labels"].shape[1]

        soft_labels = torch.zeros((len(features), max_length, num_labels), dtype=torch.float32)
        hard_labels = torch.full((len(features), max_length), -100, dtype=torch.long)
        loss_mask = torch.zeros((len(features), max_length), dtype=torch.bool)

        for feature_index, feature in enumerate(features):
            length = feature["soft_labels"].shape[0]
            soft_labels[feature_index, :length] = feature["soft_labels"]
            hard_labels[feature_index, :length] = feature["hard_labels"]
            loss_mask[feature_index, :length] = feature["loss_mask"]

        batch["soft_labels"] = soft_labels
        batch["hard_labels"] = hard_labels
        batch["loss_mask"] = loss_mask
        return batch


def distillation_loss(logits: torch.Tensor, batch: dict[str, torch.Tensor], teacher_variant: str) -> torch.Tensor:
    if teacher_variant in {"hard_argmax", "verbalized_hard_argmax"}:
        return F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            batch["hard_labels"].view(-1),
            ignore_index=-100,
        )

    log_probs = F.log_softmax(logits, dim=-1)
    token_losses = -(batch["soft_labels"] * log_probs).sum(dim=-1)
    masked_losses = token_losses[batch["loss_mask"]]
    return masked_losses.mean()


def train_distilled_model(
    model,
    train_dataloader,
    mini_eval_dataloader,
    optimizer,
    device,
    label_list: list[str],
    config: DistillConfig,
) -> dict[str, Any]:
    train_iter = iter(train_dataloader)
    global_step = 0
    total_loss = 0.0
    best_span_f1 = -1.0
    best_step = 0
    patience_left = config.patience
    early_stopped = False
    num_evals = 0
    started_at = time.time()

    pbar = tqdm(
        total=config.max_steps,
        desc=f"Distill {config.split} {config.teacher_split} {config.teacher_variant}",
        dynamic_ncols=True,
    )
    while global_step < config.max_steps:
        model.train()
        steps_this_round = (
            min(config.min_steps_before_eval - global_step, config.max_steps - global_step)
            if global_step < config.min_steps_before_eval
            else min(config.eval_every, config.max_steps - global_step)
        )

        for _ in range(steps_this_round):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)

            batch = {key: value.to(device) for key, value in batch.items()}
            model_inputs = {
                key: value
                for key, value in batch.items()
                if key not in {"soft_labels", "hard_labels", "loss_mask"}
            }

            optimizer.zero_grad()
            outputs = model(**model_inputs)
            loss = distillation_loss(outputs.logits, batch, config.teacher_variant)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            global_step += 1
            pbar.update(1)
            pbar.set_postfix_str(f"loss={loss.item():.4f} best={best_span_f1:.4f} pat={patience_left}")

        if global_step < config.min_steps_before_eval:
            continue

        num_evals += 1
        _, span_metrics = evaluate_model(model, mini_eval_dataloader, device, label_list, show_progress=False)
        span_f1 = span_metrics["Span_Strict_F1"]

        if span_f1 > best_span_f1:
            best_span_f1 = span_f1
            best_step = global_step
            patience_left = config.patience
            best_dir = config.output_dir / "best_span_f1"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
        else:
            patience_left -= 1

        pbar.set_postfix_str(f"miniSF1={span_f1:.4f} best={best_span_f1:.4f}@{best_step} pat={patience_left}")
        if patience_left <= 0:
            early_stopped = True
            break

    pbar.close()
    elapsed_seconds = time.time() - started_at
    return {
        "steps_trained": global_step,
        "avg_loss": total_loss / global_step if global_step else 0.0,
        "best_span_strict_f1": best_span_f1,
        "best_step": best_step,
        "early_stopped": early_stopped,
        "num_evals": num_evals,
        "elapsed_seconds": elapsed_seconds,
    }


def load_config_from_env() -> DistillConfig:
    split = os.environ.get("DISTILL_SPLIT", "k5_seed42")
    teacher_split = os.environ.get("DISTILL_TEACHER_SPLIT", "validation")
    teacher_variant = os.environ.get("DISTILL_TEACHER_VARIANT", "vote_normalized")
    output_dir = Path(
        os.environ.get(
            "DISTILL_OUTPUT_DIR",
            MODELS_ROOT / split / teacher_split / teacher_variant,
        )
    ).expanduser().resolve()
    return DistillConfig(
        split=split,
        teacher_split=teacher_split,
        teacher_variant=teacher_variant,
        train_file=Path(
            os.environ.get("DISTILL_TRAIN_FILE", teacher_file_for(split, teacher_split, teacher_variant))
        ).resolve(),
        mini_eval_file=Path(os.environ.get("DISTILL_MINI_EVAL_FILE", DATA_ROOT / "mini_val.jsonl")).resolve(),
        validation_file=Path(os.environ.get("DISTILL_VALIDATION_FILE", DATA_ROOT / "validation.jsonl")).resolve(),
        test_file=Path(os.environ.get("DISTILL_TEST_FILE", DATA_ROOT / "test.jsonl")).resolve(),
        metadata_file=Path(os.environ.get("DISTILL_METADATA_FILE", DATA_ROOT / "metadata.json")).resolve(),
        output_dir=output_dir,
        max_steps=int(os.environ.get("DISTILL_MAX_STEPS", MAX_TRAIN_STEPS)),
        eval_every=int(os.environ.get("DISTILL_EVAL_EVERY", EVAL_EVERY)),
        patience=int(os.environ.get("DISTILL_PATIENCE", PATIENCE)),
        min_steps_before_eval=int(os.environ.get("DISTILL_MIN_STEPS_BEFORE_EVAL", MIN_STEPS_BEFORE_EVAL)),
        train_batch_size=int(os.environ.get("DISTILL_TRAIN_BATCH_SIZE", TRAIN_BATCH_SIZE)),
        eval_batch_size=int(os.environ.get("DISTILL_EVAL_BATCH_SIZE", EVAL_BATCH_SIZE)),
        learning_rate=float(os.environ.get("DISTILL_LEARNING_RATE", LEARNING_RATE)),
        seed=int(os.environ.get("DISTILL_SEED", 42)),
        max_length=int(os.environ.get("DISTILL_MAX_LENGTH", MAX_LENGTH)),
    )


def main() -> None:
    hf_logging.set_verbosity_error()
    config = load_config_from_env()
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_json(config.metadata_file)
    label_list = [str(label) for label in metadata["label_list"]]
    label_to_id = {label: index for index, label in enumerate(label_list)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_rows = read_jsonl(config.train_file)
    train_dataset = DistillationDataset(
        train_rows,
        tokenizer,
        label_to_id,
        teacher_variant=config.teacher_variant,
        max_length=config.max_length,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=DistillationCollator(tokenizer),
    )

    tokenize_and_align_labels = tokenize_and_align_labels_fn(tokenizer, label_to_id)
    eval_dd = DatasetDict(
        {
            "mini_eval": dataset_from_jsonl(config.mini_eval_file),
            "validation": dataset_from_jsonl(config.validation_file),
            "test": dataset_from_jsonl(config.test_file),
        }
    )
    processed_eval = eval_dd.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=eval_dd["mini_eval"].column_names,
        desc="Tokenize distillation eval splits",
    )
    mini_eval_dataloader = make_dataloader(processed_eval["mini_eval"], tokenizer, config.eval_batch_size)
    validation_dataloader = make_dataloader(processed_eval["validation"], tokenizer, config.eval_batch_size)
    test_dataloader = make_dataloader(processed_eval["test"], tokenizer, config.eval_batch_size)

    model_config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print(
        "Distilled BERT run: "
        f"split={config.split} teacher_split={config.teacher_split} teacher={config.teacher_variant} "
        f"rows={len(train_rows)} max_steps={config.max_steps} device={device}"
    )
    train_summary = train_distilled_model(
        model,
        train_dataloader,
        mini_eval_dataloader,
        optimizer,
        device,
        label_list,
        config,
    )

    best_dir = config.output_dir / "best_span_f1"
    if best_dir.exists():
        model = AutoModelForTokenClassification.from_pretrained(best_dir)
        model.to(device)

    tokenizer.save_pretrained(config.output_dir)
    model.save_pretrained(config.output_dir)

    eval_results: dict[str, dict[str, float]] = {}
    for eval_name, dataloader in [
        ("mini_eval", mini_eval_dataloader),
        ("validation", validation_dataloader),
        ("test", test_dataloader),
    ]:
        token_metrics, span_metrics = evaluate_model(model, dataloader, device, label_list, show_progress=False)
        eval_results[eval_name] = {**token_metrics, **span_metrics}
        print(
            f"{eval_name}: HF F1={token_metrics['HF_Token_F1']:.4f} "
            f"Span strict F1={span_metrics['Span_Strict_F1']:.4f}"
        )

    summary = {
        "experiment": {
            "dataset": "conll2003",
            "model_name": MODEL_NAME,
            "split": config.split,
            "teacher_split": config.teacher_split,
            "teacher_variant": config.teacher_variant,
            "seed": config.seed,
            "note": (
                "Report-grade no-leakage run; teacher_train_file is unlabeled-pool-derived."
                if "unlabeled_pool" in config.teacher_split
                else "Method-validation run; current teacher_train_file is validation-derived."
            ),
        },
        "paths": {
            "train_file": str(config.train_file),
            "mini_eval_file": str(config.mini_eval_file),
            "validation_file": str(config.validation_file),
            "test_file": str(config.test_file),
            "output_dir": str(config.output_dir),
        },
        "runtime_config": {
            "max_steps": config.max_steps,
            "eval_every": config.eval_every,
            "patience": config.patience,
            "min_steps_before_eval": config.min_steps_before_eval,
            "train_batch_size": config.train_batch_size,
            "eval_batch_size": config.eval_batch_size,
            "learning_rate": config.learning_rate,
            "max_length": config.max_length,
        },
        "data_stats": {
            "train_rows": len(train_rows),
            "mini_eval_rows": len(eval_dd["mini_eval"]),
            "validation_rows": len(eval_dd["validation"]),
            "test_rows": len(eval_dd["test"]),
        },
        "train_summary": train_summary,
        "eval_results": eval_results,
    }
    write_json(config.output_dir / "training_summary.json", summary)
    print(f"Summary written to: {config.output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
