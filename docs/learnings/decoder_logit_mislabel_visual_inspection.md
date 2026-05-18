# Decoder Logit Mislabel Visual Inspection

## Why This Matters

This note records a concrete visual-inspection example from the BART soft-label pilot. It supports the decision to avoid direct decoder-logit projection as the main token-level teacher signal.

In our seq2seq setup, BART generates structured text such as `[EASTERN DIVISION]ORG`. The reliable hard-label path is to parse that generated structure and align the span back to the input tokens. The unreliable path was to project decoder vocabulary scores into BIO tags and take the token-level argmax.

## Odd Label Example

Prediction artifact:

```text
data/interim/conll2003_soft_labels/bart_base/pet_oada/k5_seed42/pattern_08_xe_improved_patterns_fixed2000/validation/hard_predictions.jsonl
line 2259
```

Raw prediction row:

```json
{
  "id": "row-002258",
  "tokens": ["EASTERN", "DIVISION"],
  "predicted_tags": ["B-ORG", "I-ORG"],
  "score_argmax_tags": ["O", "O"],
  "generated_text": "[EASTERN DIVISION]ORG",
  "ner_tags": [7, 8],
  "gold_tags": ["B-MISC", "I-MISC"]
}
```

Prompt/target row that produced the example:

```text
data/interim/conll2003_kshot_pet_oada (OLD - DELETE)/val/pattern_8_SEN_order_instr_basic.jsonl
line 2354
```

```json
{
  "input_text": "Text: EASTERN DIVISION Order: left to right Please list the entities based on the above.",
  "target_text": "[EASTERN DIVISION]MISC"
}
```

Visual-inspection interpretation:

- Gold label: `B-MISC I-MISC`
- Generated text: `[EASTERN DIVISION]ORG`
- Parsed generated BIO tags: `B-ORG I-ORG`
- Projected decoder argmax tags: `O O`

The generated text found the entity span but assigned the odd/wrong type `ORG` instead of `MISC`. The decoder-logit projection was worse for distillation, because it collapsed the generated entity to `O O`.

## Cleaner Projection Failure

There is also a cleaner case where the generated structure is correct, but the projected decoder argmax still fails.

Prediction artifact:

```text
data/interim/conll2003_soft_labels/bart_base/pet_oada/k5_seed142/pattern_02_xe_improved_patterns_fixed2000/validation/hard_predictions.jsonl
line 2259
```

Raw prediction row:

```json
{
  "id": "row-002258",
  "tokens": ["EASTERN", "DIVISION"],
  "predicted_tags": ["B-MISC", "I-MISC"],
  "score_argmax_tags": ["O", "O"],
  "generated_text": "[EASTERN DIVISION]MISC",
  "ner_tags": [7, 8],
  "gold_tags": ["B-MISC", "I-MISC"]
}
```

Prompt/target row:

```text
data/interim/conll2003_kshot_pet_oada (OLD - DELETE)/val/pattern_2_instr_SEN_order_encapsulated.jsonl
line 2354
```

```json
{
  "input_text": "For the text || EASTERN DIVISION || extract entities using the order: left to right",
  "target_text": "[EASTERN DIVISION]MISC"
}
```

This isolates the decoder-projection problem: the generated text, parsed hard labels, and gold labels all agree, but projected decoder logits still produce `O O`.

## Decision

Use parsed generated structures as the primary BART teacher output. Derive soft labels from ensemble disagreement over parsed BIO tags, not from direct projection of decoder vocabulary logits.

Appendix-ready data:

```text
reports/mikke_data/data/soft_label_failure_examples.csv
reports/mikke_data/data/soft_label_failure_examples.jsonl
```
