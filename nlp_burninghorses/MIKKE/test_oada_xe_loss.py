from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from nlp_burninghorses.MIKKE.oada_xe_loss import (
    get_oada_tau,
    make_capped_same_type_permutation_targets,
    make_same_type_permutation_targets,
    oada_xe_loss_from_logits,
    teacher_forced_oada_xe_loss_single_example,
)
from nlp_burninghorses.MIKKE.scripts.bart import oada_xe_loss_batch


class FakeBatchEncoding(dict):
    def to(self, device: torch.device | str) -> FakeBatchEncoding:
        return FakeBatchEncoding(
            {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in self.items()
            }
        )


def make_confident_logits(predicted_tokens: list[int], vocab_size: int = 8) -> torch.Tensor:
    logits = torch.full((1, len(predicted_tokens), vocab_size), -20.0)
    for position, token_id in enumerate(predicted_tokens):
        logits[0, position, token_id] = 20.0
    return logits


class FakeTokenizer:
    pad_token_id = 0

    target_ids = {
        "[Aske]PER [Mikkel]PER [ITU]ORG": [1, 2, 3],
        "[Mikkel]PER [Aske]PER [ITU]ORG": [2, 1, 3],
        "[Copenhagen]LOC": [4, 5],
    }

    def __call__(
        self,
        *,
        text_target: list[str],
        max_length: int,
        padding: bool,
        truncation: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del padding, truncation, return_tensors

        rows = [
            self.target_ids[target_text][:max_length]
            for target_text in text_target
        ]
        max_row_length = max(len(row) for row in rows)
        padded_rows = [
            row + [self.pad_token_id] * (max_row_length - len(row))
            for row in rows
        ]
        return FakeBatchEncoding({"input_ids": torch.tensor(padded_rows)})


class FakeSeq2SeqModel:
    def __init__(self, preferred_tokens: list[int]) -> None:
        self.preferred_tokens = preferred_tokens
        self.call_count = 0
        self.last_labels: torch.Tensor | None = None

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> SimpleNamespace:
        del input_ids, attention_mask
        self.call_count += 1
        self.last_labels = labels.detach().clone()

        logits = torch.full((labels.shape[0], labels.shape[1], 8), -20.0)
        for position, token_id in enumerate(self.preferred_tokens[: labels.shape[1]]):
            logits[:, position, token_id] = 20.0

        return SimpleNamespace(logits=logits)


class TestOadaXeLoss(unittest.TestCase):
    def test_target_permutations_only_swap_same_type_entities(self) -> None:
        targets = make_same_type_permutation_targets("[Aske]PER [Mikkel]PER [ITU]ORG")

        self.assertEqual(
            targets,
            [
                "[Aske]PER [Mikkel]PER [ITU]ORG",
                "[Mikkel]PER [Aske]PER [ITU]ORG",
            ],
        )

    def test_capped_permutations_keep_canonical_first_and_sample_deterministically(self) -> None:
        canonical_target = "[A]PER [B]PER [C]PER [D]PER"

        first_targets, first_total = make_capped_same_type_permutation_targets(
            canonical_target,
            max_candidate_count=12,
            seed=123,
            sample_key="example-1",
        )
        second_targets, second_total = make_capped_same_type_permutation_targets(
            canonical_target,
            max_candidate_count=12,
            seed=123,
            sample_key="example-1",
        )
        other_targets, _ = make_capped_same_type_permutation_targets(
            canonical_target,
            max_candidate_count=12,
            seed=124,
            sample_key="example-1",
        )

        self.assertEqual(first_total, 24)
        self.assertEqual(len(first_targets), 12)
        self.assertEqual(first_targets[0], canonical_target)
        self.assertEqual(first_targets, second_targets)
        self.assertNotEqual(first_targets, other_targets)
        self.assertEqual(len(set(first_targets)), len(first_targets))

    def test_capped_permutations_return_all_targets_when_under_cap(self) -> None:
        canonical_target = "[Aske]PER [Mikkel]PER [ITU]ORG"

        targets, total = make_capped_same_type_permutation_targets(
            canonical_target,
            max_candidate_count=12,
            seed=123,
        )

        self.assertEqual(total, 2)
        self.assertEqual(
            targets,
            [
                canonical_target,
                "[Mikkel]PER [Aske]PER [ITU]ORG",
            ],
        )

    def test_perfect_match_standard_xe_and_oada_xe_are_equal(self) -> None:
        logits = make_confident_logits([1, 2, 3])
        canonical_labels = torch.tensor([[1, 2, 3]])
        candidate_labels = torch.tensor([[[1, 2, 3], [2, 1, 3]]])

        result = oada_xe_loss_from_logits(
            logits=logits,
            canonical_labels=canonical_labels,
            candidate_labels=candidate_labels,
            tau=1.0,
        )

        self.assertAlmostEqual(result["normal_xe"].item(), result["oada_xe"].item(), places=6)
        self.assertLess(result["loss"].item(), 1e-5)

    def test_same_type_swap_has_low_oada_xe_but_high_standard_xe(self) -> None:
        logits = make_confident_logits([2, 1, 3])
        canonical_labels = torch.tensor([[1, 2, 3]])
        candidate_labels = torch.tensor([[[1, 2, 3], [2, 1, 3]]])

        result = oada_xe_loss_from_logits(
            logits=logits,
            canonical_labels=canonical_labels,
            candidate_labels=candidate_labels,
            tau=1.0,
        )

        self.assertGreater(result["normal_xe"].item(), 20.0)
        self.assertLess(result["oada_xe"].item(), 1e-5)
        self.assertLess(result["loss"].item(), 1e-5)

    def test_inter_type_violation_remains_high_loss(self) -> None:
        logits = make_confident_logits([3, 1, 2])
        canonical_labels = torch.tensor([[1, 2, 3]])
        candidate_labels = torch.tensor([[[1, 2, 3], [2, 1, 3]]])

        result = oada_xe_loss_from_logits(
            logits=logits,
            canonical_labels=canonical_labels,
            candidate_labels=candidate_labels,
            tau=1.0,
        )

        self.assertGreater(result["oada_xe"].item(), 20.0)
        self.assertGreater(result["loss"].item(), 20.0)

    def test_linear_annealing_schedule(self) -> None:
        logits = make_confident_logits([2, 1, 3])
        canonical_labels = torch.tensor([[1, 2, 3]])
        candidate_labels = torch.tensor([[[1, 2, 3], [2, 1, 3]]])

        start = oada_xe_loss_from_logits(logits, canonical_labels, candidate_labels, tau=0.0)
        middle = oada_xe_loss_from_logits(logits, canonical_labels, candidate_labels, tau=0.5)
        end = oada_xe_loss_from_logits(logits, canonical_labels, candidate_labels, tau=1.0)

        expected_middle = 0.5 * start["normal_xe"].item() + 0.5 * end["oada_xe"].item()

        self.assertAlmostEqual(start["loss"].item(), start["normal_xe"].item(), places=6)
        self.assertAlmostEqual(middle["loss"].item(), expected_middle, places=6)
        self.assertAlmostEqual(end["loss"].item(), end["oada_xe"].item(), places=6)
        self.assertEqual(get_oada_tau(0, warmup_steps=1000), 0.0)
        self.assertEqual(get_oada_tau(500, warmup_steps=1000), 0.5)
        self.assertEqual(get_oada_tau(1000, warmup_steps=1000), 1.0)

    def test_teacher_forced_oada_scores_each_candidate_target(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeSeq2SeqModel(preferred_tokens=[2, 1, 3])
        canonical_target = "[Aske]PER [Mikkel]PER [ITU]ORG"
        candidate_targets = make_same_type_permutation_targets(canonical_target)

        result = teacher_forced_oada_xe_loss_single_example(
            model=model,
            tokenizer=tokenizer,
            input_ids=torch.tensor([[101, 102]]),
            attention_mask=torch.tensor([[1, 1]]),
            canonical_target=canonical_target,
            candidate_targets=candidate_targets,
            tau=1.0,
            max_target_length=8,
        )

        self.assertIsNotNone(model.last_labels)
        torch.testing.assert_close(
            model.last_labels,
            torch.tensor([[1, 2, 3], [2, 1, 3]]),
        )
        self.assertGreater(result["normal_xe"].item(), 20.0)
        self.assertLess(result["oada_xe"].item(), 1e-5)
        self.assertGreater(result["oada_margin"].item(), 20.0)
        self.assertTrue(result["oada_activated"].item())
        self.assertLess(result["loss"].item(), 1e-5)

    def test_batched_oada_matches_single_example_loss_with_one_forward(self) -> None:
        tokenizer = FakeTokenizer()
        single_model = FakeSeq2SeqModel(preferred_tokens=[2, 1, 3])
        batch_model = FakeSeq2SeqModel(preferred_tokens=[2, 1, 3])
        canonical_targets = [
            "[Aske]PER [Mikkel]PER [ITU]ORG",
            "[Copenhagen]LOC",
        ]
        candidate_targets = [
            make_same_type_permutation_targets(canonical_targets[0]),
            make_same_type_permutation_targets(canonical_targets[1]),
        ]
        tau = 0.75

        single_results = [
            teacher_forced_oada_xe_loss_single_example(
                model=single_model,
                tokenizer=tokenizer,
                input_ids=torch.tensor([[101, 102]]),
                attention_mask=torch.tensor([[1, 1]]),
                canonical_target=canonical_target,
                candidate_targets=candidates,
                tau=tau,
                max_target_length=8,
            )
            for canonical_target, candidates in zip(canonical_targets, candidate_targets)
        ]
        batch_result = oada_xe_loss_batch(
            model=batch_model,
            tokenizer=tokenizer,
            batch={
                "input_ids": torch.tensor([[101, 102], [201, 202]]),
                "attention_mask": torch.tensor([[1, 1], [1, 1]]),
                "canonical_targets": canonical_targets,
                "candidate_targets": candidate_targets,
            },
            tau=tau,
            max_target_length=8,
        )

        expected_loss = torch.stack([result["loss"] for result in single_results]).mean()
        expected_normal_xe = torch.stack(
            [result["normal_xe"] for result in single_results]
        ).mean()
        expected_oada_xe = torch.stack(
            [result["oada_xe"] for result in single_results]
        ).mean()

        self.assertEqual(single_model.call_count, 2)
        self.assertEqual(batch_model.call_count, 1)
        torch.testing.assert_close(batch_result["loss"], expected_loss)
        torch.testing.assert_close(batch_result["normal_xe"], expected_normal_xe)
        torch.testing.assert_close(batch_result["oada_xe"], expected_oada_xe)
        self.assertEqual(batch_result["example_count"], 2)
        self.assertEqual(batch_result["avg_candidate_count"], 1.5)
        self.assertEqual(batch_result["max_candidate_count"], 2.0)


if __name__ == "__main__":
    unittest.main()
