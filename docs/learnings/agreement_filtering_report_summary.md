# Agreement Filtering Report Summary

## Purpose

Agreement filtering is used to convert a BART teacher ensemble into a smaller, higher-quality silver-label set for BERT distillation.

The key idea is simple:

> Keep unlabeled examples where the ensemble strongly agrees on the predicted entities, then train BERT on that filtered silver supervision.

## Filtering Strategy

For each unlabeled sentence, the BART ensemble produces token-level votes. We use the ensemble's hard prediction as the silver label and estimate confidence from how many teachers voted for each predicted token label.

The filter prioritizes sentences with predicted entities and ranks them by entity-token agreement. This means examples are preferred when the teachers agree not only that an entity exists, but also on the specific entity tokens and labels.

To avoid training only on positive/entity-heavy sentences, the filter also keeps a small capped portion of high-confidence no-entity examples. In the report setting, the selected subset is `top_1000`, with at most 10% no-entity rows.

## Why This Matters

The filtering step is not just a cleanup step. It is central to the method:

- It removes noisy or unstable pseudo-labels.
- It turns multiple BART teachers into one compact silver-label training set.
- It prevents BERT from training on the full noisy unlabeled pool.
- It makes the final method better described as agreement-filtered ensemble distillation, rather than plain majority voting.

## Reporting Language

Recommended concise wording:

> We aggregate predictions from a pattern-diverse OADA BART ensemble and select the top 1000 unlabeled examples with strongest agreement on predicted entity labels. This agreement-filtered silver set is then used to train the BERT student.

Important distinction:

- Majority vote describes how the ensemble produces a single predicted label sequence.
- Agreement filtering describes how we choose which pseudo-labeled examples are reliable enough for BERT distillation.

## Current Decision

Keep the existing agreement-filtered `top_1000` strategy for reporting the completed final scores.

Further filtering variants are useful research ideas, but they should be treated as future work or diagnostics. They are not part of the main report pipeline.
