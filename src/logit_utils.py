"""Small logit helpers for constrained decoding."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable


def log_probability(logits: list[float], token_id: int) -> float:
    """Return the log-probability of one token from raw logits."""
    if token_id < 0 or token_id >= len(logits):
        return -math.inf

    max_logit = max(logits)
    log_sum_exp = max_logit + math.log(
        sum(math.exp(logit - max_logit) for logit in logits)
    )
    return logits[token_id] - log_sum_exp


def cumulative_sequence_logprob(
    get_logits: Callable[[list[int]], list[float]],
    context_ids: list[int],
    continuation_ids: list[int],
) -> float:
    """Score a full token continuation under the current context."""
    if not continuation_ids:
        return -math.inf

    score = 0.0
    current_ids = context_ids.copy()

    for token_id in continuation_ids:
        logits = get_logits(current_ids)
        score += log_probability(logits, token_id)
        current_ids.append(token_id)

    return score


def apply_mask(logits: list[float], allowed_ids: Iterable[int]) -> list[float]:
    """Mask disallowed token ids to negative infinity."""
    masked = [-math.inf] * len(logits)

    for token_id in allowed_ids:
        if 0 <= token_id < len(logits):
            masked[token_id] = logits[token_id]

    return masked


def argmax(values: list[float]) -> int:
    """Return the index of the largest value."""
    if not values:
        raise ValueError("Cannot choose argmax from an empty sequence")
    return max(range(len(values)), key=values.__getitem__)
