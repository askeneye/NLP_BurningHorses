from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, AutoConfig, set_seed)
import torch
from torch.utils.data import DataLoader
import random
import evaluate
from tqdm.auto import tqdm


# ----------------------------------------------------------------------------
# Load data and hyperparameters
# ----------------------------------------------------------------------------

# Set random seeds
set_seed(42)

# Define hyperparameters (e.g., learning_rate, num_train_epochs, model_name)
dataset_name = "conll2003"
learning_rate = 2e-5
num_train_epochs = 3
model_name = "google-bert/bert-base-cased"

# Data percentage variable - OADA paper uses 5, 10, 20 & 50 %
data_percentage = 5 


# Load the dataset
dataset_name = "conll2003" # 
raw_datasets = load_dataset(dataset_name, trust_remote_code=True)

# ----------------------------------------------------------------------------
# Few-Shot Sampling logic
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

# REMEMBER TO CHANGE MODEL NAME BASED ON DATA PERCENTAGE VARIABLE!!!!!!
model.save_pretrained("./models/baseline_5")
tokenizer.save_pretrained("./models/baseline_5")