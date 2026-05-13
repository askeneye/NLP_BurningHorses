from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start_path: Path) -> Path:
    for path in [start_path, *start_path.parents]:
        if (path / "pyproject.toml").exists():
            return path

    raise FileNotFoundError("Could not locate project root containing pyproject.toml.")


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nlp_burninghorses.utils import span_f1  # noqa: E402

MODEL_NAME = "facebook/bart-base"
DEFAULT_DATASET = "conll2003"
DEFAULT_TASK = "kshot_seq2seq_ner"
DEFAULT_MODEL_FAMILY = "bart_base"
DEFAULT_METHOD = "pet_oada"
DEFAULT_KSHOT_SPLIT = "k5_seed242"
DEFAULT_PATTERN = "pattern_01"
DEFAULT_SOURCE_PATTERN_ID = "pattern_1_instr_SEN_order"
MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 128
MAX_GENERATION_LENGTH = 128

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_STEPS = 5000
EVAL_STEPS = 200
EARLY_STOPPING_PATIENCE = 5
MIN_STEPS_BEFORE_STOPPING = 1000

DEFAULT_TRAIN_FILE = (
    "data/interim/conll2003_kshot_seq2seq/train/pet_oada/k5_seed242/pattern_01.jsonl"
)
DEFAULT_MINI_VAL_FILE = (
    "data/interim/conll2003_kshot_seq2seq/inference/first_to_last/mini_val.jsonl"
)
DEFAULT_MINI_VAL_GOLD_FILE = "data/interim/conll2003_kshot_bert/mini_val.jsonl"
DEFAULT_VALIDATION_FILE = (
    "data/interim/conll2003_kshot_seq2seq/inference/first_to_last/validation.jsonl"
)
DEFAULT_VALIDATION_GOLD_FILE = "data/interim/conll2003_kshot_bert/validation.jsonl"
DEFAULT_METADATA_FILE = "data/interim/conll2003_kshot_bert/metadata.json"
DEFAULT_OUTPUT_DIR = (
    "models/conll2003_kshot_seq2seq/bart_base/pet_oada/k5_seed242/pattern_01"
)

BRACKETED_ENTITY_RE = re.compile(r"\[([^\]]+)\]\s*([A-Za-z][A-Za-z0-9_-]*)")


@dataclass(frozen=True)
class BartRunConfig:
    dataset: str
    task: str
    model_family: str
    method: str
    kshot_split: str
    pattern: str
    source_pattern_id: str
    model_name: str
    train_file: Path
    mini_val_file: Path
    mini_val_gold_file: Path
    validation_file: Path
    validation_gold_file: Path
    metadata_file: Path
    output_dir: Path
    max_source_length: int
    max_target_length: int
    max_generation_length: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    max_steps: int
    eval_steps: int
    early_stopping_patience: int
    min_steps_before_stopping: int
    seed: int


class JsonlSeq2SeqDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped_line = line.strip()
            if stripped_line:
                rows.append(json.loads(stripped_line))

    return rows


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def repo_relative_path(path: str | Path) -> str:
    resolved_path = Path(path).resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved_path)


def require_columns(rows: list[dict[str, Any]], required_columns: set[str], source_name: str) -> None:
    for row_index, row in enumerate(rows):
        missing_columns = required_columns.difference(row)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{source_name} row {row_index} is missing required column(s): {missing}")


def detokenize_tokens(tokens: list[str]) -> str:
    """Reconstruct readable text from CoNLL-style tokenized words."""
    no_space_before = {
        ".",
        ",",
        ":",
        ";",
        "?",
        "!",
        "%",
        ")",
        "]",
        "}",
        "'s",
        "'m",
        "'re",
        "'ve",
        "'ll",
        "'d",
        "n't",
    }
    no_space_after = {"(", "[", "{", "$", "#"}
    quote_is_open = False
    previous_opened_quote = False
    text = ""

    for token in tokens:
        if token in {'"', "``", "''"}:
            if token == "''" or (token == '"' and quote_is_open):
                text += '"'
                quote_is_open = False
                previous_opened_quote = False
            else:
                if text and not text.endswith(" "):
                    text += " "
                text += '"'
                quote_is_open = True
                previous_opened_quote = True
            continue

        if not text:
            text = token
        elif token in no_space_before or token.startswith("'"):
            text += token
        elif text[-1] in no_space_after or previous_opened_quote:
            text += token
        else:
            text += f" {token}"

        previous_opened_quote = False

    return text


def labels_to_bio_tags(label_ids: list[int | str], id_to_label: dict[int, str]) -> list[str]:
    tags: list[str] = []

    for label_id in label_ids:
        if isinstance(label_id, str) and not label_id.isdigit():
            tags.append(label_id)
        else:
            tags.append(id_to_label[int(label_id)])

    return tags


def find_entity_span(tokens: list[str], entity_text: str, occupied_indices: set[int]) -> tuple[int, int] | None:
    normalized_entity_text = entity_text.strip()
    if not normalized_entity_text:
        return None

    for start_index in range(len(tokens)):
        for end_index in range(start_index + 1, len(tokens) + 1):
            candidate_indices = set(range(start_index, end_index))
            if candidate_indices.intersection(occupied_indices):
                continue
            if detokenize_tokens(tokens[start_index:end_index]) == normalized_entity_text:
                return start_index, end_index

    return None


def generated_text_to_bio_tags(
    generated_text: str,
    tokens: list[str],
    entity_types: set[str],
) -> list[str]:
    """Convert generated bracket syntax into BIO tags aligned with source tokens."""
    tags = ["O"] * len(tokens)
    occupied_indices: set[int] = set()

    for match in BRACKETED_ENTITY_RE.finditer(generated_text):
        entity_text = match.group(1).strip()
        entity_type = match.group(2).strip()
        if entity_type not in entity_types:
            continue

        span = find_entity_span(tokens, entity_text, occupied_indices)
        if span is None:
            continue

        start_index, end_index = span
        tags[start_index] = f"B-{entity_type}"
        for index in range(start_index + 1, end_index):
            tags[index] = f"I-{entity_type}"
        occupied_indices.update(range(start_index, end_index))

    return tags


def span_metrics_from_tags(
    gold_tags_per_sentence: list[list[str]],
    pred_tags_per_sentence: list[list[str]],
) -> dict[str, float]:
    tp = fp = fn = 0
    tp_unlabeled = fp_unlabeled = fn_unlabeled = 0
    recall_loose_tp = recall_loose_fn = 0
    precision_loose_tp = precision_loose_fp = 0

    for gold_tags, pred_tags in zip(gold_tags_per_sentence, pred_tags_per_sentence):
        gold_spans = span_f1.toSpans(gold_tags)
        pred_spans = span_f1.toSpans(pred_tags)

        strict_overlap = len(gold_spans.intersection(pred_spans))
        tp += strict_overlap
        fp += len(pred_spans) - strict_overlap
        fn += len(gold_spans) - strict_overlap

        unlabeled_overlap = span_f1.getUnlabeled(gold_spans, pred_spans)
        tp_unlabeled += unlabeled_overlap
        fp_unlabeled += len(pred_spans) - unlabeled_overlap
        fn_unlabeled += len(gold_spans) - unlabeled_overlap

        recall_loose = span_f1.getLooseOverlap(gold_spans, pred_spans)
        precision_loose = span_f1.getLooseOverlap(pred_spans, gold_spans)
        recall_loose_tp += recall_loose
        recall_loose_fn += len(gold_spans) - recall_loose
        precision_loose_tp += precision_loose
        precision_loose_fp += len(pred_spans) - precision_loose

    strict_precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    strict_recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    strict_f1 = (
        0.0
        if strict_precision + strict_recall == 0
        else 2 * strict_precision * strict_recall / (strict_precision + strict_recall)
    )

    unlabeled_precision = (
        0.0 if tp_unlabeled + fp_unlabeled == 0 else tp_unlabeled / (tp_unlabeled + fp_unlabeled)
    )
    unlabeled_recall = (
        0.0 if tp_unlabeled + fn_unlabeled == 0 else tp_unlabeled / (tp_unlabeled + fn_unlabeled)
    )
    unlabeled_f1 = (
        0.0
        if unlabeled_precision + unlabeled_recall == 0
        else 2 * unlabeled_precision * unlabeled_recall / (unlabeled_precision + unlabeled_recall)
    )

    loose_precision = (
        0.0
        if precision_loose_tp + precision_loose_fp == 0
        else precision_loose_tp / (precision_loose_tp + precision_loose_fp)
    )
    loose_recall = (
        0.0
        if recall_loose_tp + recall_loose_fn == 0
        else recall_loose_tp / (recall_loose_tp + recall_loose_fn)
    )
    loose_f1 = (
        0.0
        if loose_precision + loose_recall == 0
        else 2 * loose_precision * loose_recall / (loose_precision + loose_recall)
    )

    return {
        "Span_Strict_Precision": strict_precision,
        "Span_Strict_Recall": strict_recall,
        "Span_Strict_F1": strict_f1,
        "Span_Unlabeled_Precision": unlabeled_precision,
        "Span_Unlabeled_Recall": unlabeled_recall,
        "Span_Unlabeled_F1": unlabeled_f1,
        "Span_Loose_Precision": loose_precision,
        "Span_Loose_Recall": loose_recall,
        "Span_Loose_F1": loose_f1,
    }


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


def evaluate_bart_generation(
    model,
    tokenizer,
    eval_rows: list[dict[str, Any]],
    eval_gold_rows: list[dict[str, Any]],
    id_to_label: dict[int, str],
    entity_types: set[str],
    config: BartRunConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    gold_tags_per_sentence: list[list[str]] = []
    pred_tags_per_sentence: list[list[str]] = []

    for batch_start in range(0, len(eval_rows), config.eval_batch_size):
        batch = eval_rows[batch_start : batch_start + config.eval_batch_size]
        gold_batch = eval_gold_rows[batch_start : batch_start + config.eval_batch_size]
        input_texts = [row["input_text"] for row in batch]
        tokenized = tokenizer(
            input_texts,
            max_length=config.max_source_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                max_length=config.max_generation_length,
            )

        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for gold_row, generated_text in zip(gold_batch, generated_texts):
            tokens = gold_row["tokens"]
            gold_tags_per_sentence.append(labels_to_bio_tags(gold_row["ner_tags"], id_to_label))
            pred_tags_per_sentence.append(
                generated_text_to_bio_tags(generated_text, tokens, entity_types)
            )

    return span_metrics_from_tags(gold_tags_per_sentence, pred_tags_per_sentence)


def train_one_bart_model(config: BartRunConfig) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    training_started_at = time.time()

    train_rows = read_jsonl(config.train_file)
    mini_val_rows = read_jsonl(config.mini_val_file)
    mini_val_gold_rows = read_jsonl(config.mini_val_gold_file)
    validation_rows = read_jsonl(config.validation_file)
    validation_gold_rows = read_jsonl(config.validation_gold_file)
    metadata = read_json(config.metadata_file)

    require_columns(train_rows, {"input_text", "target_text"}, str(config.train_file))
    require_columns(mini_val_rows, {"input_text", "target_text"}, str(config.mini_val_file))
    require_columns(validation_rows, {"input_text", "target_text"}, str(config.validation_file))
    require_columns(
        mini_val_gold_rows,
        {"tokens", "ner_tags"},
        str(config.mini_val_gold_file),
    )
    require_columns(
        validation_gold_rows,
        {"tokens", "ner_tags"},
        str(config.validation_gold_file),
    )
    if len(mini_val_rows) != len(mini_val_gold_rows):
        raise ValueError(
            "Mini-val generated rows and raw gold rows must have the same length: "
            f"{len(mini_val_rows)} != {len(mini_val_gold_rows)}"
        )
    if len(validation_rows) != len(validation_gold_rows):
        raise ValueError(
            "Validation generated rows and raw gold rows must have the same length: "
            f"{len(validation_rows)} != {len(validation_gold_rows)}"
        )

    label_list = metadata["label_list"]
    id_to_label = {int(label_id): label for label_id, label in metadata["id_to_label"].items()}
    entity_types = {label[2:] for label in label_list if label != "O" and "-" in label}

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = JsonlSeq2SeqDataset(train_rows)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=make_seq2seq_collate_fn(
            tokenizer,
            config.max_source_length,
            config.max_target_length,
        ),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = config.output_dir / "best_span_f1"
    eval_history_path = config.output_dir / "eval_history.jsonl"
    eval_history_path.unlink(missing_ok=True)

    train_iterator = iter(train_dataloader)
    global_step = 0
    total_loss = 0.0
    last_loss = 0.0
    best_span_f1 = -1.0
    best_step = 0
    patience_left = config.early_stopping_patience
    early_stopped = False
    num_evals = 0
    eval_history: list[dict[str, Any]] = []

    progress_bar = tqdm(total=config.max_steps, desc="BART train", dynamic_ncols=True)

    while global_step < config.max_steps:
        model.train()

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        last_loss = loss.item()
        total_loss += last_loss
        global_step += 1

        progress_bar.update(1)
        progress_bar.set_postfix_str(
            f"loss={last_loss:.3f} best={best_span_f1:.3f}@{best_step} pat={patience_left}"
        )

        if global_step % config.eval_steps != 0:
            continue

        num_evals += 1
        span_metrics = evaluate_bart_generation(
            model=model,
            tokenizer=tokenizer,
            eval_rows=mini_val_rows,
            eval_gold_rows=mini_val_gold_rows,
            id_to_label=id_to_label,
            entity_types=entity_types,
            config=config,
            device=device,
        )
        span_f1_score = span_metrics["Span_Strict_F1"]
        is_new_best = span_f1_score > best_span_f1

        if is_new_best:
            best_span_f1 = span_f1_score
            best_step = global_step
            patience_left = config.early_stopping_patience
            best_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
        elif global_step >= config.min_steps_before_stopping:
            patience_left -= 1

        eval_record = {
            "step": global_step,
            "eval_index": num_evals,
            "last_loss": last_loss,
            "avg_loss_so_far": total_loss / global_step if global_step else 0.0,
            **span_metrics,
            "best_span_strict_f1": best_span_f1,
            "best_step": best_step,
            "patience_left": patience_left,
            "is_new_best": is_new_best,
        }
        eval_history.append(eval_record)
        with eval_history_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(eval_record, ensure_ascii=False) + "\n")

        progress_bar.set_postfix_str(
            f"loss={last_loss:.3f} miniSF1={span_f1_score:.3f} "
            f"best={best_span_f1:.3f}@{best_step} pat={patience_left}"
        )

        if global_step >= config.min_steps_before_stopping and patience_left <= 0:
            early_stopped = True
            break

    progress_bar.close()

    if best_step == 0:
        best_model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)
        best_step = global_step

    final_model = AutoModelForSeq2SeqLM.from_pretrained(best_model_dir)
    final_model.to(device)
    final_metrics = evaluate_bart_generation(
        model=final_model,
        tokenizer=tokenizer,
        eval_rows=mini_val_rows,
        eval_gold_rows=mini_val_gold_rows,
        id_to_label=id_to_label,
        entity_types=entity_types,
        config=config,
        device=device,
    )
    validation_metrics = evaluate_bart_generation(
        model=final_model,
        tokenizer=tokenizer,
        eval_rows=validation_rows,
        eval_gold_rows=validation_gold_rows,
        id_to_label=id_to_label,
        entity_types=entity_types,
        config=config,
        device=device,
    )
    total_training_seconds = time.time() - training_started_at

    summary = {
        "model_name": config.model_name,
        "experiment": {
            "dataset": config.dataset,
            "task": config.task,
            "model_family": config.model_family,
            "hf_model_name": config.model_name,
            "method": config.method,
            "kshot_split": config.kshot_split,
            "pattern": config.pattern,
            "source_pattern_id": config.source_pattern_id,
            "seed": config.seed,
        },
        "train_file": repo_relative_path(config.train_file),
        "mini_val_file": repo_relative_path(config.mini_val_file),
        "mini_val_gold_file": repo_relative_path(config.mini_val_gold_file),
        "validation_file": repo_relative_path(config.validation_file),
        "validation_gold_file": repo_relative_path(config.validation_gold_file),
        "metadata_file": repo_relative_path(config.metadata_file),
        "output_dir": repo_relative_path(config.output_dir),
        "data_stats": {
            "train_rows": len(train_rows),
            "mini_val_rows": len(mini_val_rows),
            "mini_val_gold_rows": len(mini_val_gold_rows),
            "validation_rows": len(validation_rows),
            "validation_gold_rows": len(validation_gold_rows),
        },
        "steps_trained": global_step,
        "total_training_seconds": total_training_seconds,
        "total_training_minutes": total_training_seconds / 60,
        "avg_loss": total_loss / global_step if global_step else 0.0,
        "best_step": best_step,
        "best_span_strict_f1": best_span_f1,
        "early_stopped": early_stopped,
        "num_evals": num_evals,
        "eval_history_file": repo_relative_path(eval_history_path),
        "eval_history": eval_history,
        "final_best_model_mini_val_metrics": final_metrics,
        "final_best_model_validation_metrics": validation_metrics,
    }

    with (config.output_dir / "training_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, ensure_ascii=False)

    return summary


def _path_from_env(name: str, default_relative_path: str) -> Path:
    value = os.environ.get(name)
    if value is not None:
        return Path(value).expanduser().resolve()

    return PROJECT_ROOT / default_relative_path


def load_config_from_env() -> BartRunConfig:
    return BartRunConfig(
        dataset=os.environ.get("BART_DATASET", DEFAULT_DATASET),
        task=os.environ.get("BART_TASK", DEFAULT_TASK),
        model_family=os.environ.get("BART_MODEL_FAMILY", DEFAULT_MODEL_FAMILY),
        method=os.environ.get("BART_METHOD", DEFAULT_METHOD),
        kshot_split=os.environ.get("BART_KSHOT_SPLIT", DEFAULT_KSHOT_SPLIT),
        pattern=os.environ.get("BART_PATTERN", DEFAULT_PATTERN),
        source_pattern_id=os.environ.get("BART_SOURCE_PATTERN_ID", DEFAULT_SOURCE_PATTERN_ID),
        model_name=os.environ.get("BART_MODEL_NAME", MODEL_NAME),
        train_file=_path_from_env("BART_TRAIN_FILE", DEFAULT_TRAIN_FILE),
        mini_val_file=_path_from_env("BART_MINI_VAL_FILE", DEFAULT_MINI_VAL_FILE),
        mini_val_gold_file=_path_from_env(
            "BART_MINI_VAL_GOLD_FILE",
            DEFAULT_MINI_VAL_GOLD_FILE,
        ),
        validation_file=_path_from_env("BART_VALIDATION_FILE", DEFAULT_VALIDATION_FILE),
        validation_gold_file=_path_from_env(
            "BART_VALIDATION_GOLD_FILE",
            DEFAULT_VALIDATION_GOLD_FILE,
        ),
        metadata_file=_path_from_env("BART_METADATA_FILE", DEFAULT_METADATA_FILE),
        output_dir=_path_from_env("BART_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        max_source_length=int(os.environ.get("BART_MAX_SOURCE_LENGTH", MAX_SOURCE_LENGTH)),
        max_target_length=int(os.environ.get("BART_MAX_TARGET_LENGTH", MAX_TARGET_LENGTH)),
        max_generation_length=int(os.environ.get("BART_MAX_GENERATION_LENGTH", MAX_GENERATION_LENGTH)),
        train_batch_size=int(os.environ.get("BART_TRAIN_BATCH_SIZE", TRAIN_BATCH_SIZE)),
        eval_batch_size=int(os.environ.get("BART_EVAL_BATCH_SIZE", EVAL_BATCH_SIZE)),
        learning_rate=float(os.environ.get("BART_LEARNING_RATE", LEARNING_RATE)),
        max_steps=int(os.environ.get("BART_MAX_STEPS", MAX_STEPS)),
        eval_steps=int(os.environ.get("BART_EVAL_STEPS", EVAL_STEPS)),
        early_stopping_patience=int(
            os.environ.get("BART_EARLY_STOPPING_PATIENCE", EARLY_STOPPING_PATIENCE)
        ),
        min_steps_before_stopping=int(
            os.environ.get("BART_MIN_STEPS_BEFORE_STOPPING", MIN_STEPS_BEFORE_STOPPING)
        ),
        seed=int(os.environ.get("BART_SEED", 42)),
    )


def main() -> None:
    config = load_config_from_env()
    summary = train_one_bart_model(config)
    print(
        "Finished BART training: "
        f"steps={summary['steps_trained']} "
        f"best_step={summary['best_step']} "
        f"best_mini_span_f1={summary['best_span_strict_f1']:.4f} "
        f"time_min={summary['total_training_minutes']:.2f}"
    )
    print(f"Summary written to: {config.output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
