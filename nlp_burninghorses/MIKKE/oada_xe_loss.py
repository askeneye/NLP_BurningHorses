from __future__ import annotations

from collections import defaultdict
import hashlib
from itertools import permutations
import random
import re
from typing import Any

import torch
import torch.nn.functional as F

TARGET_ENTITY_RE = re.compile(r"\[([^\]]+)\]\s*([A-Za-z][A-Za-z0-9_-]*)")


def parse_target_text(target_text: str) -> list[dict[str, str]]:
    """Parse BART NER targets formatted as '[entity]TYPE [entity]TYPE'."""
    entities: list[dict[str, str]] = []

    for match in TARGET_ENTITY_RE.finditer(target_text):
        entities.append(
            {
                "text": match.group(1).strip(),
                "type": match.group(2).strip(),
            }
        )

    return entities


def format_entities_as_target(entities: list[dict[str, str]], empty_target: str = "") -> str:
    """Format parsed entities back into the seq2seq target representation."""
    if not entities:
        return empty_target

    return " ".join(f"[{entity['text']}]{entity['type']}" for entity in entities)


def make_same_type_permutation_targets(target_text: str, empty_target: str = "") -> list[str]:
    """
    Generate target strings that preserve entity-type order but relax same-type order.

    Example:
        [A]PER [B]PER [C]LOC -> [A]PER [B]PER [C]LOC and [B]PER [A]PER [C]LOC
    """
    entities = parse_target_text(target_text)
    if not entities:
        return [empty_target]

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    type_order: list[str] = []

    for entity in entities:
        entity_type = entity["type"]
        if entity_type not in grouped:
            type_order.append(entity_type)
        grouped[entity_type].append(entity)

    per_type_options: list[list[tuple[dict[str, str], ...]]] = []
    for entity_type in type_order:
        same_type_entities = grouped[entity_type]
        if len(same_type_entities) <= 1:
            per_type_options.append([tuple(same_type_entities)])
        else:
            per_type_options.append(list(permutations(same_type_entities)))

    targets: list[str] = []

    def backtrack(index: int, current: list[dict[str, str]]) -> None:
        if index == len(per_type_options):
            targets.append(format_entities_as_target(current, empty_target=empty_target))
            return

        for option in per_type_options[index]:
            backtrack(index + 1, current + list(option))

    backtrack(0, [])
    return sorted(set(targets))


def make_capped_same_type_permutation_targets(
    target_text: str,
    max_candidate_count: int | None = None,
    seed: int = 42,
    empty_target: str = "",
    sample_key: str | None = None,
) -> tuple[list[str], int]:
    """
    Generate same-type permutation targets with deterministic candidate capping.

    The canonical target is always placed first. If a cap is active, the remaining
    candidates are sampled from valid non-canonical permutations using a stable
    per-example seed so DataLoader order and epoch count do not change the subset.
    Returns the used candidates and the uncapped candidate count.
    """
    all_targets = make_same_type_permutation_targets(target_text, empty_target=empty_target)
    non_canonical_targets = [target for target in all_targets if target != target_text]
    ordered_targets = [target_text, *non_canonical_targets]
    total_candidate_count = len(ordered_targets)

    if (
        max_candidate_count is None
        or max_candidate_count <= 0
        or total_candidate_count <= max_candidate_count
    ):
        return ordered_targets, total_candidate_count

    if max_candidate_count == 1:
        return [target_text], total_candidate_count

    stable_key = sample_key if sample_key is not None else target_text
    seed_material = f"{seed}\0{stable_key}".encode("utf-8")
    stable_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    rng = random.Random(stable_seed)
    sampled_targets = rng.sample(non_canonical_targets, max_candidate_count - 1)
    return [target_text, *sampled_targets], total_candidate_count


def sequence_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute mean cross-entropy per sequence.

    Returns one scalar per batch item so OADA-XE can choose the best candidate per example.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch_size, sequence_length, vocab_size]")
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch_size, sequence_length]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must agree on batch and sequence dimensions")

    vocab_size = logits.shape[-1]
    token_losses = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(labels.shape)

    valid_tokens = labels.ne(ignore_index)
    token_counts = valid_tokens.sum(dim=1).clamp_min(1)
    return (token_losses * valid_tokens).sum(dim=1) / token_counts


def oada_xe_loss_from_logits(
    logits: torch.Tensor,
    canonical_labels: torch.Tensor,
    candidate_labels: torch.Tensor,
    tau: float,
    ignore_index: int = -100,
) -> dict[str, torch.Tensor]:
    """
    Blend normal XE with the lowest-loss valid OADA candidate.

    Args:
        logits: Tensor shaped [batch_size, sequence_length, vocab_size].
        canonical_labels: Tensor shaped [batch_size, sequence_length].
        candidate_labels: Tensor shaped [batch_size, num_candidates, sequence_length].
        tau: Annealing factor where 0.0 is pure XE and 1.0 is pure OADA-XE.
    """
    if candidate_labels.ndim != 3:
        raise ValueError(
            "candidate_labels must have shape [batch_size, num_candidates, sequence_length]"
        )
    if candidate_labels.shape[0] != canonical_labels.shape[0]:
        raise ValueError("candidate_labels and canonical_labels must have the same batch size")
    if candidate_labels.shape[2] != canonical_labels.shape[1]:
        raise ValueError("candidate labels must have the same sequence length as canonical labels")

    clamped_tau = float(max(0.0, min(1.0, tau)))
    batch_size, num_candidates, sequence_length = candidate_labels.shape

    normal_xe_per_example = sequence_cross_entropy(
        logits=logits,
        labels=canonical_labels,
        ignore_index=ignore_index,
    )

    expanded_logits = (
        logits.unsqueeze(1)
        .expand(batch_size, num_candidates, sequence_length, logits.shape[-1])
        .reshape(batch_size * num_candidates, sequence_length, logits.shape[-1])
    )
    flattened_candidates = candidate_labels.reshape(batch_size * num_candidates, sequence_length)
    candidate_losses = sequence_cross_entropy(
        logits=expanded_logits,
        labels=flattened_candidates,
        ignore_index=ignore_index,
    ).reshape(batch_size, num_candidates)

    oada_xe_per_example = candidate_losses.min(dim=1).values
    blended_per_example = (
        (1.0 - clamped_tau) * normal_xe_per_example
        + clamped_tau * oada_xe_per_example
    )

    return {
        "loss": blended_per_example.mean(),
        "normal_xe": normal_xe_per_example.mean(),
        "oada_xe": oada_xe_per_example.mean(),
        "tau": torch.tensor(clamped_tau, device=logits.device, dtype=logits.dtype),
    }


def tokenize_target_texts(
    tokenizer: Any,
    target_texts: list[str],
    max_target_length: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Tokenize target strings and mask pad tokens for seq2seq loss computation."""
    if not target_texts:
        raise ValueError("target_texts must not be empty")

    tokenized = tokenizer(
        text_target=target_texts,
        max_length=max_target_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenized["input_ids"]
    labels[labels == tokenizer.pad_token_id] = ignore_index
    return labels


def teacher_forced_candidate_losses(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_texts: list[str],
    max_target_length: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Score candidate targets with seq2seq teacher forcing.

    The source tensors must contain exactly one example. They are repeated once per
    candidate target so every target gets its own decoder-side conditioning.
    """
    if input_ids.shape[0] != 1 or attention_mask.shape[0] != 1:
        raise ValueError("teacher_forced_candidate_losses expects a single source example")

    labels = tokenize_target_texts(
        tokenizer=tokenizer,
        target_texts=target_texts,
        max_target_length=max_target_length,
        ignore_index=ignore_index,
    ).to(input_ids.device)
    candidate_count = labels.shape[0]

    outputs = model(
        input_ids=input_ids.expand(candidate_count, -1),
        attention_mask=attention_mask.expand(candidate_count, -1),
        labels=labels,
    )

    return sequence_cross_entropy(
        logits=outputs.logits,
        labels=labels,
        ignore_index=ignore_index,
    )


def teacher_forced_oada_xe_loss_single_example(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    canonical_target: str,
    candidate_targets: list[str],
    tau: float,
    max_target_length: int,
    ignore_index: int = -100,
) -> dict[str, torch.Tensor]:
    """
    Compute annealed OADA-XE for one seq2seq source example.

    Unlike oada_xe_loss_from_logits, this function is suitable for BART training:
    each candidate target is scored under its own teacher-forced decoder history.
    """
    all_targets = [canonical_target] + [
        target for target in candidate_targets if target != canonical_target
    ]

    candidate_losses = teacher_forced_candidate_losses(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_texts=all_targets,
        max_target_length=max_target_length,
        ignore_index=ignore_index,
    )

    clamped_tau = float(max(0.0, min(1.0, tau)))
    normal_xe = candidate_losses[0]
    oada_xe = candidate_losses.min()
    oada_margin = normal_xe - oada_xe
    oada_activated = oada_margin > 1e-6
    loss = (1.0 - clamped_tau) * normal_xe + clamped_tau * oada_xe

    return {
        "loss": loss,
        "normal_xe": normal_xe.detach(),
        "oada_xe": oada_xe.detach(),
        "oada_margin": oada_margin.detach(),
        "oada_activated": oada_activated.detach(),
        "tau": torch.tensor(clamped_tau, device=input_ids.device, dtype=loss.dtype),
    }


def get_oada_tau(
    global_step: int,
    tau_start: float = 0.0,
    tau_end: float = 1.0,
    warmup_steps: int = 1000,
) -> float:
    """Linearly anneal tau from tau_start to tau_end over warmup_steps."""
    if warmup_steps <= 0:
        return tau_end

    progress = min(max(global_step, 0) / warmup_steps, 1.0)
    return tau_start + progress * (tau_end - tau_start)


def tokenize_candidate_targets(
    tokenizer: Any,
    candidate_targets: list[list[str]],
    max_target_length: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Tokenize ragged candidate target strings into [batch, candidates, seq_len].

    This helper is intended for integration tests and BART training collation.
    """
    if not candidate_targets:
        raise ValueError("candidate_targets must not be empty")

    candidate_count = max(len(targets) for targets in candidate_targets)
    if candidate_count == 0:
        raise ValueError("each example must have at least one candidate target")

    padded_targets = [
        targets + [targets[0]] * (candidate_count - len(targets))
        for targets in candidate_targets
    ]
    flat_targets = [target for targets in padded_targets for target in targets]

    labels = tokenize_target_texts(
        tokenizer=tokenizer,
        target_texts=flat_targets,
        max_target_length=max_target_length,
        ignore_index=ignore_index,
    )

    return labels.reshape(len(candidate_targets), candidate_count, labels.shape[-1])
