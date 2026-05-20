# OADA Ladder Interpretation

## Context

The OADA paper reports that adding OADA-style ordering to BART improves few-shot CoNLL-2003 NER. Their reported BART+OADA scores are:

- `k=5`: `47.96`
- `k=10`: `58.06`
- `k=20`: `65.20`
- `k=50`: `73.04`

Our closest matching single-teacher result is `OADA BART (single teacher)`, which uses canonical verbalized labels and the original OADA-style prompt:

```text
Sentence: {SEN} Following the order: {PERM} Entity:
```

Our scores are:

- `k=5`: `47.02`
- `k=10`: `53.85`
- `k=20`: `61.76`
- `k=50`: `72.44`

This broadly matches the reported BART+OADA pattern, especially at `k=5` and `k=50`, and supports using the single OADA BART teacher as the report baseline for the OADA-style BART stage.

## Interpretation

The OADA paper's ablation is useful evidence that OADA-style BART teachers are stronger than raw BART input. However, their ladder does not fully isolate permutation augmentation from the prompt interface needed to expose the ordering information to a seq2seq model.

In our experiments, removing OADA permutations while keeping a structured prompt ensemble had little effect after distillation:

- `Pat+OADA BART ensemble -> BERT`: average `68.51`
- `Pat BART ensemble -> BERT`: average `68.68`
- difference: `-0.16` F1

This suggests that the final gains in our setup are not cleanly attributable to permutation augmentation alone. The stronger interpretation is that OADA-style BART teachers are useful inputs to a silver-label pipeline, and the largest practical gains come from teacher selection, agreement/confidence filtering, ensembling, and BERT distillation.

## Main Ablation Ladder

For the report, use a simple procedural ladder rather than a claim of fully isolated causal components:

1. `BERT`
2. `OADA BART single teacher`
3. `OADA BART single-teacher confidence -> BERT`
4. `OADA BART ensemble agreement -> BERT`

Recommended wording:

> This ladder is a procedural ablation, not a factorial decomposition. It shows how performance changes when moving from a gold-only BERT baseline to an OADA-style BART teacher, then to confidence-filtered single-teacher distillation, and finally to agreement-filtered ensemble distillation.

## Reporting Guidance

Use humble, factual language:

- We build on prior evidence that OADA-style prompting/augmentation improves BART few-shot NER.
- Our single OADA BART scores are comparable to the OADA paper's BART+OADA row.
- Our additional ablations suggest that permutation augmentation alone is not the dominant explanation for the final BERT gains.
- The final method should be described as agreement-filtered distillation from an OADA BART ensemble, not as a pure OADA effect.

Avoid implying intent or overclaiming about the original paper. The safe statement is:

> The original OADA ablation does not fully disentangle permutation augmentation from the ordering-aware prompt interface. Our results suggest that, in the BART-to-BERT distillation setting, structured prompting and silver-label filtering are major contributors to the observed gains.
