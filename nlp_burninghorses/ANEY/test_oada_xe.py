import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from oada_xe import (
    make_same_type_permutation_targets,
    seq2seq_ce_loss_for_targets,
)

model_name = "facebook/bart-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()

input_text = (
    'List all entities in this sentence: The Briton, who lost '
    'his World Boxing Council (WBC) title to Mike Tyson...'
)

target_text = (
    "[Briton]MISC "
    "[World Boxing Council]ORG "
    "[WBC]ORG "
    "[Mike Tyson]PER"
)

targets = make_same_type_permutation_targets(target_text)

print("\nEquivalent OADA targets:")
for target in targets:
    print("  ", target)

losses = seq2seq_ce_loss_for_targets(
    model=model,
    tokenizer=tokenizer,
    input_text=input_text,
    target_texts=targets,
    device=device,
)

print("\nCross-entropy loss for each equivalent target:")
for target, loss in sorted(losses, key=lambda x: x[1]):
    print(f"{loss:.4f} | {target}")

print("\nNormal XE would train against only one target.")
print("OADA-XE chooses the lowest-loss equivalent target.")