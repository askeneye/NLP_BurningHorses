# OADA-XE Pilot Findings And Final Pattern Design Notes

## Purpose

This note summarizes the completed BART-base CoNLL2003 k-shot seq2seq NER experiments so the results can be cross-examined against the original OADA/PET methodology and paper scores.

The main conclusion is that the completed runs are useful proof-of-concept evidence, but they should not be treated as final paper-aligned OADA runs because the original OADA prompt was mis-specified.

## Prompt Correction

The completed `pattern_01` runs used:

```text
Sentence: {SEN} Order: {PERM}
```

The original OADA prompt should be:

```text
Sentence: {SEN} Following the order: {PERM} Entity:
```

This change matters because the corrected prompt is more than a cosmetic edit. It adds:

- A natural-language order cue: `Following the order:`
- An explicit generation cue: `Entity:`
- A sentence-first layout that matches the strongest observed pattern family

All completed models should therefore be interpreted as a pilot over the OADA-XE objective and candidate policy, not as final comparable OADA paper reproductions.

## Completed Experiment Inventory

The completed comparison matrix contains:

- `pattern_01_xe_fixed2000`: 12 runs, all k/seeds.
- `pattern_01_oada_xe_cap24_fixed2000`: 12 runs, all k/seeds.
- `pattern_01_oada_xe_cap12_fixed3600`: 12 prior OADA reference runs, all k/seeds.
- Full-pattern OADA-XE pilot for two splits:
  - `k5_seed42`, patterns 01-12.
  - `k10_seed42`, patterns 01-12.

The primary fixed-step policy was:

```text
max_steps = 2000
eval_steps = 100
model_selection_strategy = final_step
use_early_stopping = false
oada_tau_warmup_steps = 2000
oada_candidate_cap = 24
oada_candidate_seed = 42
BART_SEED = 42
train_batch_size = 8
```

## XE vs OADA-XE Final Validation F1

The following table compares final validation strict F1 for the directly comparable fixed-2000 `pattern_01` runs.

| Split | XE fixed-2000 | OADA-XE cap24 fixed-2000 | Delta |
|---|---:|---:|---:|
| k5_seed42 | 0.4655 | 0.5069 | +0.0414 |
| k5_seed142 | 0.3605 | 0.2857 | -0.0748 |
| k5_seed242 | 0.4041 | 0.3946 | -0.0095 |
| k10_seed42 | 0.5254 | 0.5411 | +0.0157 |
| k10_seed142 | 0.5233 | 0.5253 | +0.0020 |
| k10_seed242 | 0.4558 | 0.4032 | -0.0527 |
| k20_seed42 | 0.5833 | 0.5914 | +0.0081 |
| k20_seed142 | 0.5535 | 0.5493 | -0.0041 |
| k20_seed242 | 0.5828 | 0.5849 | +0.0021 |
| k50_seed42 | 0.6995 | 0.7266 | +0.0270 |
| k50_seed142 | 0.7210 | 0.7167 | -0.0043 |
| k50_seed242 | 0.7090 | 0.7191 | +0.0102 |

Aggregate by k:

| k | XE mean | OADA-XE mean | Mean delta | OADA wins |
|---|---:|---:|---:|---:|
| k5 | 0.4100 | 0.3957 | -0.0143 | 1/3 |
| k10 | 0.5015 | 0.4899 | -0.0117 | 2/3 |
| k20 | 0.5732 | 0.5752 | +0.0020 | 2/3 |
| k50 | 0.7098 | 0.7208 | +0.0110 | 2/3 |

Overall:

- OADA-XE wins 7 out of 12 paired runs.
- Mean final-F1 delta is approximately -0.0033.
- OADA-XE is negative at k5 and k10 under the old prompt.
- OADA-XE becomes positive at k20 and k50.

Interpretation:

> Under the old prompt, OADA-XE is not uniformly better than XE, but it becomes more favorable as k increases.

This is consistent with the idea that the OADA objective may need enough examples for permutation robustness to help rather than inject noise.

## OADA-XE Cap24 Fixed-2000 vs OADA-XE Cap12 Fixed-3600

The cap24 fixed-2000 run was compared against the earlier cap12 fixed-3600 run.

Mean final-F1 delta by k, computed as cap24 fixed-2000 minus cap12 fixed-3600:

| k | Mean delta | Interpretation |
|---|---:|---|
| k5 | -0.0189 | Longer cap12 run was better on average |
| k10 | -0.0021 | Essentially tied |
| k20 | -0.0162 | Longer cap12 run was better on average |
| k50 | +0.0074 | Shorter cap24 run was better on average |

Interpretation:

> The 2000-step cap24 setup is not obviously inferior at high k and is much cheaper, but lower-k results remain unstable.

This supports using 2000 steps for a corrected-prompt pilot before deciding whether final runs need 1800, 2000, 2200, or longer.

## Step Count And Fitness Findings

Across fixed-2000 runs, many models peaked before the final step on mini-val strict F1.

OADA-XE best mini-val steps:

| k | Best steps across seeds | Mean best-final gap |
|---|---|---:|
| k5 | 1800, 1500, 1700 | 0.0337 |
| k10 | 1700, 1200, 500 | 0.0577 |
| k20 | 600, 1400, 1600 | 0.0643 |
| k50 | 1200, 1800, 1000 | 0.0202 |

XE best mini-val steps:

| k | Best steps across seeds | Mean best-final gap |
|---|---|---:|
| k5 | 1300, 1000, 1200 | 0.0572 |
| k10 | 300, 1500, 400 | 0.0557 |
| k20 | 1700, 1400, 1600 | 0.0551 |
| k50 | 1100, 1900, 1000 | 0.0336 |

Interpretation:

- `2000` steps is plausible as a fixed budget.
- It is not universally optimal.
- Low-k runs are noisy and often peak earlier.
- k50 is more stable and less sensitive to the exact final step.

Recommendation:

> Keep 2000 steps for the corrected-prompt pilot. Recheck fitness curves before launching the full final matrix.

## Candidate Cap And Candidate Pressure

The cap24 policy handled large candidate spaces without crashing or excessive runtime.

Important candidate pressure cases:

| Split | Max used/total candidates | Capped evals | OADA-XE delta vs XE |
|---|---:|---:|---:|
| k5_seed42 | 24/48 | 14 | +0.0414 |
| k50_seed42 | 24/288 | 4 | +0.0270 |
| k50_seed242 | 24/40320 | 3 | +0.0102 |

Interpretation:

- Cap 24 is useful and practical.
- Extreme candidate spaces exist even in CoNLL2003.
- Candidate pressure alone does not explain all gains.
- The cap preserves deterministic runtime while allowing OADA-XE to search meaningful alternatives.

Recommendation:

> Keep `oada_candidate_cap = 24` and `oada_candidate_seed = 42` for final runs.

## Full-Pattern Pilot Findings

A 12-pattern OADA-XE cap24 fixed-2000 pilot was completed for `k5_seed42` and `k10_seed42`.

Per-pattern final validation strict F1:

| Pattern | k5_seed42 | k10_seed42 |
|---|---:|---:|
| p01 | 0.5069 | 0.5411 |
| p02 | 0.4742 | 0.5236 |
| p03 | 0.4616 | 0.5523 |
| p04 | 0.4070 | 0.5290 |
| p05 | 0.4419 | 0.4931 |
| p06 | 0.4447 | 0.5142 |
| p07 | 0.4188 | 0.5115 |
| p08 | 0.4713 | 0.5067 |
| p09 | 0.3677 | 0.5205 |
| p10 | 0.2537 | 0.4884 |
| p11 | 0.4728 | 0.5504 |
| p12 | 0.2816 | 0.4618 |

Pattern summary:

| Split | Mean | Standard deviation | Best | Worst |
|---|---:|---:|---|---|
| k5_seed42 | 0.4169 | 0.0788 | p01 = 0.5069 | p10 = 0.2537 |
| k10_seed42 | 0.5161 | 0.0265 | p03 = 0.5523 | p12 = 0.4618 |

Interpretation:

- Pattern choice matters strongly at k5.
- Pattern choice still matters at k10, but variance is smaller.
- Some prompt forms are consistently risky.
- Ensemble diversity is useful, but diversity should come from natural-language formulation rather than odd symbolic wrappers.

## Traits Of Weak Patterns

The weakest patterns were:

```text
p10: [ Order: {PERM} ] Extract entities from the text: {SEN}
p12: [ {PERM} ] || Text: {SEN} || Extract the entities.
```

Common weak traits:

- Order comes before the sentence.
- The order is wrapped in square brackets.
- The prompt is terse.
- The sentence appears after the ordering instruction.
- The output cue is generic or absent.

Important nuance:

`p07` also used brackets, but it placed the sentence first:

```text
{SEN} || [ Order: {PERM} ] || Extract the entities from the previous text following the specified sequence.
```

It was not among the two worst patterns. Therefore the best conclusion is not simply that brackets are always bad.

Better conclusion:

> Terse, order-first, bracketed prompts are risky. Brackets become especially harmful when `{PERM}` is isolated before the sentence.

## Traits Of Stronger Patterns

The stronger patterns tended to have one or more of the following:

- Sentence/text appears early.
- The sentence is explicitly labeled.
- The order is introduced in natural language.
- The task is explicitly entity extraction or entity listing.
- The prompt has a clear output cue.
- The prompt avoids square brackets around `{PERM}`.

Examples of stronger observed patterns:

```text
p01: Sentence: {SEN} Order: {PERM}
p03: Extract the entities using the following order: {PERM} || Text: {SEN}
p11: Order: {PERM} || {SEN} || Based on the requested order and the text provided, list the entities.
```

`p11` is useful diagnostically because it shows that order-first is not automatically bad. A clear natural-language explanation can rescue order-first layouts.

## Pattern Count Strategy

Original PET methodology used:

- 10 patterns for k <= 10.
- 5 patterns for k > 10.

This fits the current evidence well:

- Low-k runs benefit from prompt diversity, but bad prompts can hurt sharply.
- Higher-k runs can use a smaller, safer prompt set.
- The first five patterns should therefore be the safest core.
- Patterns 6-10 should provide natural-language diversity without symbolic wrappers.

## Recommended New Pattern Bank

Design principles:

- Use the corrected original OADA prompt as `pattern_01`.
- Keep all prompts natural-language based.
- Remove bracketed `{PERM}` wrappers.
- Avoid bare `{PERM}` at the beginning.
- Add an explicit output cue such as `Entity:` or `Entities:`.
- Order the first five as the recommended core set for k > 10.
- Use all ten for k <= 10.

Recommended 10-pattern family:

```yaml
patterns:
  - id: "pattern_1_oada_original"
    template: "Sentence: {SEN} Following the order: {PERM} Entity:"
    type: "sentence-order-entity-cue"

  - id: "pattern_2_text_order_entities"
    template: "Text: {SEN} Following the entity order: {PERM} Entities:"
    type: "sentence-order-entity-cue"

  - id: "pattern_3_extract_sentence_order"
    template: "Extract entities from the sentence: {SEN} Follow this order: {PERM} Entities:"
    type: "instruction-sentence-order"

  - id: "pattern_4_order_sentence_entity"
    template: "Following the order: {PERM}, extract entities from this sentence: {SEN} Entity:"
    type: "order-sentence-entity-cue"

  - id: "pattern_5_sentence_list_order"
    template: "Sentence: {SEN} List the entities following this order: {PERM} Entities:"
    type: "sentence-instruction-order"

  - id: "pattern_6_given_text_order"
    template: "Given the text: {SEN} Use the order {PERM} to list the entities. Entities:"
    type: "sentence-instruction-order"

  - id: "pattern_7_entity_order_then_sentence"
    template: "Entity order: {PERM}. Sentence: {SEN} Extract the entities:"
    type: "order-sentence-entity-cue"

  - id: "pattern_8_sentence_entities_in_order"
    template: "Sentence: {SEN} Entities in the following order, {PERM}:"
    type: "sentence-order-entity-cue"

  - id: "pattern_9_find_entities_order"
    template: "Find the entities in this text: {SEN} Use this order: {PERM} Entities:"
    type: "instruction-sentence-order"

  - id: "pattern_10_order_guided_extraction"
    template: "Use the entity order {PERM} for the sentence: {SEN} Entity:"
    type: "order-sentence-entity-cue"
```

## Recommended Final-Run Sequence

Before launching the full final matrix:

1. Regenerate train and inference data for the corrected pattern bank.
2. Run a corrected-prompt pilot:
   - `k5_seed42`
   - `k10_seed42`
   - optionally `k50_seed42`
3. Compare XE vs OADA-XE again under corrected `pattern_01`.
4. Recheck fitness curves around 2000 steps.
5. If stable, launch final runs:
   - 10 patterns for k5 and k10.
   - 5 patterns for k20 and k50.
   - OADA-XE cap24 fixed-2000 unless pilot fitness suggests a change.

## Final Working Hypotheses

1. The original paper-style prompt should improve alignment with OADA.
2. OADA-XE is most promising at higher k.
3. Low-k instability is real and prompt-sensitive.
4. Cap 24 is a reasonable runtime-quality tradeoff.
5. Natural-language diversity is preferable to symbolic prompt diversity.
6. Final models should not include the bracketed order-first prompt family.
