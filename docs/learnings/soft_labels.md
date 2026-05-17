# Soft Labels

## Context

The original PET formulation uses masked language modeling, where label-word probabilities can be read directly from the masked token distribution. Our BART setup adapts PET/OADA to sequence-to-sequence NER, where the model emits structured text such as:

```text
[EASTERN DIVISION]ORG
```

This means BART does not naturally produce token-classification logits for each input token. Any token-level soft labels for a BERT student must therefore be derived from generated structures, not assumed to exist directly.

## Concrete Failure Case

For this validation row:

```json
{
  "tokens": ["EASTERN", "DIVISION"],
  "predicted_tags": ["O", "O"],
  "generated_text": "[EASTERN DIVISION]ORG",
  "ner_tags": [7, 8],
  "gold_tags": ["B-MISC", "I-MISC"]
}
```

The intuitive hard prediction from the generated text is:

```text
["B-ORG", "I-ORG"]
```

The model found the correct span text, but assigned the wrong entity type compared with gold:

```text
gold:      B-MISC I-MISC
generated: B-ORG  I-ORG
```

However, our first soft-label extraction attempt computed `predicted_tags` by taking the argmax over projected decoder-token scores. In this case, fallback scores for non-candidate labels tied or exceeded the projected `ORG` scores, so the argmax collapsed to:

```text
["O", "O"]
```

That is not the hard prediction we want for comparison. It is an artifact of trying to map seq2seq decoder scores into token-classification labels.

## Lesson

For seq2seq BART teachers, hard labels should be obtained by parsing the generated bracketed structure and aligning the generated spans back to the input tokens.

Decoder-score projection is fragile because generation-step vocabulary logits do not map cleanly to per-token BIO distributions. It can produce misleading hard labels and should not be treated as the primary distillation signal.

## Decision

Use BART ensemble members as structured hard-label teachers first, then derive soft labels from ensemble disagreement.

Recommended pipeline:

1. Each BART member generates bracketed NER output.
2. Parse each member output into token-level BIO tags, e.g. `[EASTERN DIVISION]ORG` becomes `B-ORG I-ORG`.
3. For each token, collect hard BIO tags from all ensemble members.
4. Convert member votes into a soft BIO distribution.
5. Train the BERT student on the ensemble vote distribution.

Example with 10 members:

```text
B-ORG:  6 votes
B-MISC: 3 votes
O:      1 vote
```

becomes:

```text
P(B-ORG)=0.6, P(B-MISC)=0.3, P(O)=0.1
```

This gives a meaningful soft label for the student without relying on questionable decoder-logit projection.

## Artifact Policy

Keep separate artifacts:

- Per-member hard predictions for comparison, debugging, and ensemble-vote construction.
- Ensemble-level soft labels for BERT distillation.
- Optional ensemble argmax hard labels for a hard-pseudo-label BERT baseline.

The soft-label training target should be the ensemble vote distribution, not the argmax of projected decoder logits.
