# BART Ensemble to BERT Distillation Strategy

## Research Direction

Working research question:

> Can a distilled BERT NER model improve by learning from BART teacher ensembles that combine PET pattern diversity and OADA ordering?

This is not the final paper headline, but it captures the experiment logic. The core idea is to test whether structured seq2seq teachers can create better supervision for a parallel token-classification student.

## Current Decisions

- Stop using OADA-XE loss for report-grade runs. The completed pilots did not show consistent benefit over standard XE.
- Keep OADA-style ordering augmentation as a data/ordering mechanism, not as the OADA-XE objective.
- Use careful report terminology: `PET-style prompt pattern diversity` and `OADA-style ordering augmentation`. The project borrows useful mechanisms from PET and OADA, but does not reproduce full PET PVP/MLM training or the full OADA objective.
- Treat BART seq2seq generation as a structured hard-label teacher.
- Build student soft labels from ensemble disagreement/votes, not from projected decoder logits.
- Keep hard predictions and ensemble soft labels as separate artifacts.
- Update this document whenever the experiment plan changes.

## Teacher Ensemble Ablations

To isolate what helps, compare ensembles that differ by one source of diversity at a time.

### 1. Single BART Teacher

One BART-base model trained with one prompt pattern and one seed.

Purpose:

- Establish the single-teacher seq2seq baseline.
- Compare against ensemble variants.

### 2. Seed Ensemble Without PET Diversity

Equal-size ensemble where every member uses the same non-OADA prompt, but each member has a different model seed.

Purpose:

- Measure the value of ensembling and seed diversity alone.
- Provide the fair comparator for PET pattern diversity.

### 3. PET Pattern Ensemble Without OADA Ordering

Equal-size ensemble with multiple natural-language PET prompts, but no ordering instruction and no permutation augmentation.

Purpose:

- Measure the contribution of PET pattern diversity alone.

### 4. OADA Ordering Ensemble

Permutation-augmented data with an ordering instruction, but without PET pattern diversity if used as a clean ablation.

Purpose:

- Measure the contribution of OADA ordering/permutation augmentation.

### 5. PET + OADA Ensemble

The full teacher setup: natural-language pattern diversity plus OADA ordering/permutation augmentation.

Purpose:

- Main BART teacher ensemble for distillation.

## Non-OADA Data Policy

For non-OADA ensembles, avoid the `{PERM}` slot entirely.

Do not use `"first to last"` during training for non-OADA ablations, because it still introduces an ordering instruction and contaminates the comparison.

Non-OADA training should use:

- one row per sentence,
- canonical left-to-right target order,
- prompts with no ordering phrase.

Example:

```text
Sentence: {SEN} Entity:
```

Target:

```text
[Mikkel]PER [Oslo]LOC
```

OADA training should use:

- multiple rows per sentence,
- one row per entity-type permutation,
- target ordered by the requested entity-type order,
- prompts containing `{PERM}`.

Example:

```text
Sentence: {SEN} Following the order: {PERM} Entity:
```

## Distillation Label Policy

The first projected soft-label attempt is not the final method. It tried to map BART decoder vocabulary scores into token-level BIO distributions over the original coarse labels, but this proved fragile.

Use this instead:

1. Each BART member generates bracketed output.
2. Parse generated spans into token-level BIO tags.
3. Save per-member hard predictions for comparison and debugging.
4. Aggregate per-member hard predictions into ensemble vote distributions.
5. Train BERT on the ensemble vote distributions.

Example:

```text
Member votes for token 1:
B-ORG:  6
B-MISC: 3
O:      1
```

Soft target:

```text
P(B-ORG)=0.6, P(B-MISC)=0.3, P(O)=0.1
```

Generate three teacher variants from the same member-vote counts:

- `vote_normalized`: flat normalized vote probabilities. This is the primary teacher target.
- `vote_temp2`: temperature-smoothed vote probabilities using `p(label)^(1/T) / sum_j p(j)^(1/T)` with `T=2`.
- `hard_argmax`: majority-vote hard pseudo labels for comparison.

Do not treat member-vote counts as logits. A direct `softmax(counts / T)` is less appropriate here because the BART ensemble produces categorical votes, not calibrated MLM logits.

## Verbalized Label Pilot

A promising side path is to replace terse label strings with semantic verbalizers in the BART target format:

- `PER -> person`
- `LOC -> location`
- `ORG -> organization`
- `MISC -> other`

Tokenizer verification for `facebook/bart-base` showed that `ORG` and `MISC` split into multiple tokens, while the verbalizers have single-token forms after the bracket separator. The pilot therefore uses targets such as:

```text
[Mikkel] person [Copenhagen] location
```

This makes the decoder type token semantically meaningful and easier to probe. A contained `k5_seed42`, `pattern_01`, 1000-step XE pilot reached validation strict F1 `0.5179`, so it is competitive enough to keep investigating but not yet a replacement for the full PET/OADA ensemble.

The first verbalizer-logit probe over 200 validation rows found:

- 485 decoded entities,
- 466 entities with one of the four canonical verbalized labels,
- 100% agreement between the verbalizer-probability argmax and the decoded surface label for those valid entities,
- mean probability assigned to the decoded verbalized label of about `0.955`.

Important implementation detail: force `num_beams=1` when collecting decoder-step scores for this probe. With beam search, `generated.scores` rows do not align directly with the returned sequence row, which can make the emitted label and score distribution appear contradictory.

This reopens a plausible route to true soft labels, but only for generated spans. The next step is to map verbalizer-token probabilities back to BIO labels for generated entities and compare them against ensemble-vote soft labels.

Matched 2000-step verbalized runs later strengthened the case for verbalized labels as a report-relevant setting. Across 30 matched CoNLL BART runs against the non-verbalized `xe_improved_patterns_fixed2000` setting, verbalization improved selected-model validation strict F1 in 24/30 runs, with mean delta about `+0.0295` and median delta about `+0.0287`. The effect was strongest for k10 and k20, mixed but positive for k5, and nearly neutral for k50. This should be captured in the report as evidence that semantic label surfaces help most in the lower-data regime.

## Appendix-Worthy Pilot Findings

Some small pilots did not become the main path, but they shaped the final design and are useful to document. Treat items with active follow-up data as provisional and revisit their wording before the final report:

- OADA-XE loss did not show consistent improvement over standard XE. Keep OADA as an ordering/data augmentation mechanism, but do not spend report-grade compute on the OADA-XE objective.
- Projected decoder-logit soft labels over terse labels were fragile. The concrete failure mode was that generated spans could be parsed correctly while projected class logits produced misleading token argmax labels.
- Ensemble-vote teachers were more robust than projected decoder logits. The useful variants are `hard_argmax`, `vote_normalized`, and `vote_temp2`, with hard argmax often stronger in early no-leakage BERT distillation.
- Verbalized true-soft labels are technically feasible and decoder probabilities look coherent, but the first 3-member BERT student underperformed. Shelf this for redesign rather than scaling the first implementation; revisit if stronger verbalized teachers change the result.
- Using the full unlabeled pool gives maximum reuse, but 14k pseudo-labeled sentences may be excessive for few-shot distillation. Agreement-filtered subsets should be explored before treating the full pool as the default student training set; revisit after subset distillation runs.
- The 1000-step BART pilots were useful for screening, but final BART comparisons should use matched 2000-step settings because many 1000-step runs peaked at or near the final step.
- Ensemble diversity should be quantified in the appendix. Candidate statistics include token-level vote entropy, pairwise member disagreement, span-level Jaccard/F1 between members, entity-type confusion diversity, and comparisons between PET-pattern ensembles and seed-only ensembles; tune conclusions after these diagnostics are computed.
- Verbalizer diversity is a possible mini-study. Since semantic label surfaces helped, try alternative surface forms such as `person`/`human`, `organization`/`company`, `location`/`place`, and `other`/`miscellaneous` to test whether gains come from verbalization in general, specific lexical choices, or added ensemble diversity.

## Immediate Next Step

Reset the obsolete projected soft-label artifacts and build the end-to-end method on the smallest useful scope:

1. Regenerate hard predictions for the `k=5` splits only.
2. Convert hard predictions into ensemble vote soft labels.
3. Train a distilled BERT student on the k=5 ensemble vote labels.
4. Compare against:
   - standard k-shot BERT,
   - single BART teacher,
   - seed-only BART ensemble,
   - PET/OADA BART ensemble,
   - distilled BERT from hard argmax labels,
   - distilled BERT from vote-soft labels.

Once the k=5 pipeline works end to end, expand to k=10, k=20, and k=50.
