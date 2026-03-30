# English EWT (Universal NER / UD-style IOB2): train on data/raw/en_ewt-ud-train.iob2, dev on en_ewt-ud-dev.iob2.
# From repo root: ./venv/bin/python nlp_burninghorses/BERT_Baseline_en_ewt_ud.py
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
import os
import csv
from transformers import (AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, AutoConfig, set_seed)
import torch
from torch.utils.data import DataLoader
import random
import evaluate
from tqdm.auto import tqdm
import span_f1


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_en_ewt_ud_iob2(path: str):
    """Read Universal NER IOB2: tab lines index, token, NER, …; # comments and blank lines separate sentences."""
    sentences_tokens, sentences_tags, label_set = [], [], set()
    current_tokens, current_tags = [], []

    def flush():
        nonlocal current_tokens, current_tags
        if current_tokens:
            sentences_tokens.append(current_tokens)
            sentences_tags.append(current_tags)
            current_tokens, current_tags = [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                flush()
                continue
            if not line.strip():
                flush()
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            current_tokens.append(parts[1])
            current_tags.append(parts[2])
            label_set.add(parts[2])
        flush()
    return sentences_tokens, sentences_tags, label_set


def load_en_ewt_ud_iob2_datasets(root: str) -> DatasetDict:
    train_path = os.path.join(root, "data", "raw", "en_ewt-ud-train.iob2")
    dev_path = os.path.join(root, "data", "raw", "en_ewt-ud-dev.iob2")
    tr_tok, tr_tag, s_train = parse_en_ewt_ud_iob2(train_path)
    dv_tok, dv_tag, s_dev = parse_en_ewt_ud_iob2(dev_path)
    label_names = sorted(s_train | s_dev)
    if "O" in label_names:
        label_names.remove("O")
        label_list = ["O"] + label_names
    else:
        label_list = label_names
    label_to_id = {l: i for i, l in enumerate(label_list)}

    def encode_tags(tag_seqs):
        return [[label_to_id[t] for t in sent] for sent in tag_seqs]

    features = Features(
        {
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=label_list)),
        }
    )
    train_ds = Dataset.from_dict(
        {"tokens": tr_tok, "ner_tags": encode_tags(tr_tag)},
        features=features,
    )
    val_ds = Dataset.from_dict(
        {"tokens": dv_tok, "ner_tags": encode_tags(dv_tag)},
        features=features,
    )
    return DatasetDict(train=train_ds, validation=val_ds)


# ----------------------------------------------------------------------------
# Load data and hyperparameters
# ----------------------------------------------------------------------------

# Set random seeds
set_seed(42)

# Define hyperparameters
learning_rate = 2e-5
num_train_epochs = 3
model_name = "google-bert/bert-base-cased"

# Data percentage variable - OADA paper uses 10 %
data_percentage = 100


# Load the dataset (local IOB2 files under data/raw/)
raw_datasets = load_en_ewt_ud_iob2_datasets(_repo_root())

results_filename = "BERT_en_ewt_ud_full_results.csv"

# ----------------------------------------------------------------------------
# Data percentage logic
# ----------------------------------------------------------------------------

# Calculate the requested percentage
total_train_examples = len(raw_datasets["train"])
num_few_shot_examples = int(total_train_examples * (data_percentage / 100.0))

print(f"Total original training examples: {total_train_examples}")
print(f"Sampling {data_percentage}% of the data: {num_few_shot_examples} examples")

# Shuffle the training data (using the fixed seed for reproducibility) and select the top N examples
few_shot_train = raw_datasets["train"].shuffle(seed=42).select(range(num_few_shot_examples))

# Overwrite the original train set in the dataset dictionary
raw_datasets["train"] = few_shot_train

# ----------------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------------

# Identify Text and Label Columns
# For CoNLL-2003, the text column is typically tokens and the label column is ner_tags.
# Identify columns in raw_datasets['train'].features
text_column_name = "tokens"
label_column_name = "ner_tags"

# Build a label list which will be useful for computing metrics using seqeval.
features = raw_datasets["train"].features
label_list = features[label_column_name].feature.names

# Prepare the Tokenizer and Align Labels
# We need a tokenizer that handles subwords, then align each token with the correct label (or -100 for tokens that don’t align to a label).

# a) Fill in the code to ignore label_ids after the line elif word_id == prev_word_id:
# Load the tokenizer and model config
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for i, label in enumerate(label_list)}

config = AutoConfig.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id_to_label,
    label2id=label_to_id
)

def tokenize_and_align_labels(examples):
    """
    For each example, tokenize the list of tokens and align the original labels 
    to the resulting subwords. Tokens can be split into multiple subwords, so we mark 
    the "extra" subwords with -100 to ignore them in the loss.
    """

    # 1) Tokenize
    # 'is_split_into_words=True' tells the tokenizer each item in the list is already a separate word/token.
    tokenized_inputs = tokenizer(
        examples[text_column_name],  # e.g., examples["tokens"]
        max_length=128,             
        padding=False,              
        truncation=True, 
        is_split_into_words=True
    )

    # 2) Prepare a new "labels" list aligned to the subword tokens
    all_labels = []
    
    # examples[label_column_name] might look like: [0, 0, 1, 2, ...] for each token
    for batch_index, labels in enumerate(examples[label_column_name]):
        # 'word_ids()' returns a list the same length as the subword-tokens,
        # each entry telling you which 'word' or token it came from
        word_ids = tokenized_inputs.word_ids(batch_index=batch_index)

        label_ids = []
        prev_word_id = None
        
        for word_id in word_ids:
            if word_id is None:
                # e.g. special tokens or padding
                label_ids.append(-100)
            elif word_id == prev_word_id:
                # subword token of the same word => ignore 
                label_ids.append(-100)
            else:
                # new subword, so use the label for the original token
                label_ids.append(labels[word_id])
            
            prev_word_id = word_id
        
        all_labels.append(label_ids)

    # 3) Attach the new "labels" to our tokenized inputs
    tokenized_inputs["labels"] = all_labels

    # 4) Return the updated dictionary
    return tokenized_inputs

# ----------------------------------------------------------------------------
# Process the Dataset
# ----------------------------------------------------------------------------

# Apply the tokenization and label alignment function, removing original columns to keep only model inputs.
# Map the tokenize_and_align_labels function to the raw datasets
processed_raw_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
    desc="Running tokenizer on dataset"
)

train_dataset = processed_raw_datasets["train"]
eval_dataset = processed_raw_datasets["validation"]

# Inspect a few training samples after tokenization
for index in random.sample(range(len(train_dataset)), 3):
    print(f"Sample {index} of the training set: {train_dataset[index]}")

# Define the Model and Data Collator
# Set up the token classification model using the config, and create a data collator that can handle token classification tasks.
# Initialize the model with AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    config=config
)

# Create a data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Create DataLoaders
# Use the processed dataset and data collator to build PyTorch DataLoader objects.
# Create train and eval dataloaders
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=8)
eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=8)

# Optimizer
# Initialize an optimizer.
# Move model to device (CPU/GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Create optimizer (e.g. AdamW)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ----------------------------------------------------------------------------
# Training Loop
# ----------------------------------------------------------------------------

for epoch in range(num_train_epochs):
    model.train()
    total_loss = 0

    progress_bar = tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc=f"Training Epoch {epoch+1}"
    )

    for step, batch in progress_bar:
        # Move batch to device (CPU/GPU)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass (model computes loss when labels are provided)
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Track total loss
        total_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

    # Compute average loss for the epoch
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} - Average training loss: {avg_loss:.4f}")


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

# Define Metric Calculation
# For example, use seqeval to evaluate precision, recall, and F1 on the named entity labels.
# Load seqeval metric
metric = evaluate.load("seqeval")

# a) Define utility function get_labels(predictions, references)
def get_labels(predictions, references):
    true_predictions = []
    true_labels = []

    for pred_seq, ref_seq in zip(predictions, references):
        pred_labels = []
        ref_labels = []

        for pred_id, ref_id in zip(pred_seq, ref_seq):
            ref_id = ref_id.item()
            pred_id = pred_id.item()

            if ref_id != -100:
                pred_labels.append(label_list[pred_id])
                ref_labels.append(label_list[ref_id])

        true_predictions.append(pred_labels)
        true_labels.append(ref_labels)

    return true_predictions, true_labels


def compute_metrics(preds, refs):
    results = metric.compute(predictions=preds, references=refs)
    return {
        "Precision": results["overall_precision"],
        "Recall": results["overall_recall"],
        "F1": results["overall_f1"],
        "Accuracy": results["overall_accuracy"],
    }

# Evaluation
# Switch to eval mode, run inference, and measure performance with the chosen metric.
# After training, evaluate on the validation set
model.eval()
validation_progress_bar = tqdm(range(len(eval_dataloader)))
all_predictions = []
all_labels = []
for step, batch in enumerate(eval_dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    predictions = outputs.logits.argmax(dim=-1)
    labels = batch["labels"]
    predicted_labels, true_labels = get_labels(predictions, labels)
    all_predictions.extend(predicted_labels)
    all_labels.extend(true_labels)
    validation_progress_bar.update(1)

validation_metrics = compute_metrics(all_predictions, all_labels)
validation_metrics


# ----------------------------------------------------------------------------
# Save model
# ----------------------------------------------------------------------------

model_dir = f"./models/en_ewt_ud_baseline_{data_percentage}"
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f"Model saved to {model_dir}")


# =====================================================================
# Format data for Span f1 
# =====================================================================

def export_for_span_f1(dataset, predictions, true_labels, gold_file="gold.txt", pred_file="pred.txt"):
    """
    Writes the gold and predicted labels to tab-separated text files 
    compatible with the span_f1.py script.
    """
    print(f"Exporting results to {gold_file} and {pred_file}...")
    
    with open(gold_file, "w", encoding="utf-8") as fg, open(pred_file, "w", encoding="utf-8") as fp:
        for i in range(len(dataset)):
            # Get original words from the dataset
            tokens = dataset[i][text_column_name] 
            
            # Get the aligned labels for this sentence
            sentence_true_tags = true_labels[i]
            sentence_pred_tags = predictions[i]
            
            # Sanity check: ensure truncation hasn't caused mismatched lengths
            min_len = min(len(tokens), len(sentence_true_tags), len(sentence_pred_tags))
            tokens = tokens[:min_len]
            sentence_true_tags = sentence_true_tags[:min_len]
            sentence_pred_tags = sentence_pred_tags[:min_len]
            
            for token, true_tag, pred_tag in zip(tokens, sentence_true_tags, sentence_pred_tags):
                # The script expects the tag at index 2. 
                # Format: [Word] \t [DummyColumn] \t [Tag]
                fg.write(f"{token}\t-\t{true_tag}\n")
                fp.write(f"{token}\t-\t{pred_tag}\n")
            
            # Add an empty line to signal the end of the sentence
            fg.write("\n")
            fp.write("\n")




# =====================================================================
# Export results
# =====================================================================

# Define target directory 
output_dir = f"data/processed/en_ewt_ud_baseline_{data_percentage}"
os.makedirs(output_dir, exist_ok=True)

gold_file_path = os.path.join(output_dir, "gold.txt")
pred_file_path = os.path.join(output_dir, "pred.txt")

export_for_span_f1(
    dataset=raw_datasets["validation"], 
    predictions=all_predictions, 
    true_labels=all_labels,
    gold_file=gold_file_path,
    pred_file=pred_file_path
)

print("Calculating span metrics natively...")

# Use the module you imported at the top of your script!
gold_ners = span_f1.readNlu(gold_file_path)
pred_ners = span_f1.readNlu(pred_file_path)

tp = 0; fp = 0; fn = 0
recall_loose_tp = 0; recall_loose_fn = 0
precision_loose_tp = 0; precision_loose_fp = 0
tp_ul = 0; fp_ul = 0; fn_ul = 0 

for gold_ner, pred_ner in zip(gold_ners, pred_ners):
    gold_spans = span_f1.toSpans(gold_ner)
    pred_spans = span_f1.toSpans(pred_ner)
    
    # Strict
    overlap = len(gold_spans.intersection(pred_spans))
    tp += overlap
    fp += len(pred_spans) - overlap
    fn += len(gold_spans) - overlap
    
    # Unlabeled
    overlap_ul = span_f1.getUnlabeled(gold_spans, pred_spans)
    tp_ul += overlap_ul
    fp_ul += len(pred_spans) - overlap_ul
    fn_ul += len(gold_spans) - overlap_ul

    # Loose
    overlap_loose_rec = span_f1.getLooseOverlap(gold_spans, pred_spans)
    recall_loose_tp += overlap_loose_rec
    recall_loose_fn += len(gold_spans) - overlap_loose_rec

    overlap_loose_prec = span_f1.getLooseOverlap(pred_spans, gold_spans)
    precision_loose_tp += overlap_loose_prec
    precision_loose_fp += len(pred_spans) - overlap_loose_prec

# Calculate final percentages
prec = 0.0 if tp+fp == 0 else tp/(tp+fp)
rec = 0.0 if tp+fn == 0 else tp/(tp+fn)
strict_f1 = 0.0 if prec+rec == 0.0 else 2 * (prec * rec) / (prec + rec)

prec_ul = 0.0 if tp_ul+fp_ul == 0 else tp_ul/(tp_ul+fp_ul)
rec_ul = 0.0 if tp_ul+fn_ul == 0 else tp_ul/(tp_ul+fn_ul)
ul_f1 = 0.0 if prec_ul+rec_ul == 0.0 else 2 * (prec_ul * rec_ul) / (prec_ul + rec_ul)

prec_l = 0.0 if precision_loose_tp + precision_loose_fp == 0 else precision_loose_tp/(precision_loose_tp+precision_loose_fp)
rec_l = 0.0 if recall_loose_tp+recall_loose_fn == 0 else recall_loose_tp/(recall_loose_tp+recall_loose_fn)
loose_f1 = 0.0 if prec_l+rec_l == 0.0 else 2 * (prec_l * rec_l) / (prec_l + rec_l)

print(f"Strict F1: {strict_f1:.4f} | Unlabeled F1: {ul_f1:.4f} | Loose F1: {loose_f1:.4f}")

# Combine Hugging Face token metrics and the Span metrics into one row
combined_metrics = {
    "Data_Percentage": data_percentage,
    
    # Hugging Face Token-Level Metrics (from seqeval)
    "HF_Token_Precision": validation_metrics["Precision"],
    "HF_Token_Recall": validation_metrics["Recall"],
    "HF_Token_F1": validation_metrics["F1"],
    "HF_Token_Accuracy": validation_metrics["Accuracy"],
    
    # Span-Level Strict Metrics
    "Span_Strict_Precision": prec,
    "Span_Strict_Recall": rec,
    "Span_Strict_F1": strict_f1,
    
    # Span-Level Unlabeled Metrics
    "Span_Unlabeled_Precision": prec_ul,
    "Span_Unlabeled_Recall": rec_ul,
    "Span_Unlabeled_F1": ul_f1,
    
    # Span-Level Loose Metrics
    "Span_Loose_Precision": prec_l,
    "Span_Loose_Recall": rec_l,
    "Span_Loose_F1": loose_f1
}

# Append above row to master CSV file
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)
csv_file_path = os.path.join(reports_dir, results_filename)

# Check if file exists
file_exists = os.path.isfile(csv_file_path)

with open(csv_file_path, mode="a", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=combined_metrics.keys())
    
    if not file_exists:
        writer.writeheader()  # Write column names on the very first run
        
    writer.writerow(combined_metrics)

print(f"Metrics successfully appended to {csv_file_path}")