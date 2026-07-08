from __future__ import annotations

import math
from typing import Any

from src.errors import SelectionError
from src.tokenizer_utils import encode_to_ids


EncodedCandidate = tuple[list[int], Any]


def choose_constrained_candidate(
    model: Any,
    context_ids: list[int],
    candidates: list[tuple[str, Any]],
) -> tuple[float, Any]:
    """Choose the best candidate using constrained token scoring.

    At each token position, scoring is limited to next tokens that keep at
    least one encoded candidate possible.
    """
    if not candidates:
        raise SelectionError("No candidates available")

    encoded_candidates = _encode_candidates(model, candidates)

    scored_candidates = [
        (
            _score_candidate(
                model=model,
                context_ids=context_ids,
                candidate_ids=candidate_ids,
                all_candidates=encoded_candidates,
            ),
            original_value,
        )
        for candidate_ids, original_value in encoded_candidates
    ]

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    return scored_candidates[0]


def _encode_candidates(
    model: Any,
    candidates: list[tuple[str, Any]],
) -> list[EncodedCandidate]:
    """Encode candidate texts and keep their original return values."""
    encoded_candidates: list[EncodedCandidate] = []

    for candidate_text, original_value in candidates:
        candidate_ids = encode_to_ids(model, candidate_text)
        if not candidate_ids:
            raise SelectionError("Candidate encoded to an empty token list")
        encoded_candidates.append((candidate_ids, original_value))

    return encoded_candidates


def _score_candidate(
    model: Any,
    context_ids: list[int],
    candidate_ids: list[int],
    all_candidates: list[EncodedCandidate],
) -> float:
    """Score one encoded candidate with constrained next-token choices."""
    score = 0.0
    current_ids = context_ids.copy()

    for position, token_id in enumerate(candidate_ids):
        prefix = candidate_ids[:position]
        allowed_token_ids = _allowed_next_tokens(prefix, all_candidates)

        if token_id not in allowed_token_ids:
            raise SelectionError("Candidate token is not allowed by prefix")

        logits = model.get_logits_from_input_ids(current_ids)
        score += _constrained_log_probability(
            logits=logits,
            token_id=token_id,
            allowed_token_ids=allowed_token_ids,
        )
        current_ids.append(token_id)

    return score


def _allowed_next_tokens(
    prefix: list[int],
    candidates: list[EncodedCandidate],
) -> set[int]:
    """Return next tokens for candidates that share the generated prefix."""
    allowed_token_ids: set[int] = set()

    for candidate_ids, _ in candidates:
        if len(candidate_ids) <= len(prefix):
            continue
        if candidate_ids[: len(prefix)] == prefix:
            allowed_token_ids.add(candidate_ids[len(prefix)])

    if not allowed_token_ids:
        raise SelectionError("No allowed next tokens for candidate prefix")

    return allowed_token_ids


def _constrained_log_probability(
    logits: list[float],
    token_id: int,
    allowed_token_ids: set[int],
) -> float:
    """Compute log-probability over allowed token ids only."""
    _validate_token_ids(logits, allowed_token_ids)

    max_logit = max(logits[token] for token in allowed_token_ids)
    log_sum_exp = max_logit + math.log(
        sum(math.exp(logits[token] - max_logit) for token in allowed_token_ids)
    )
    return logits[token_id] - log_sum_exp


def _validate_token_ids(
    logits: list[float],
    token_ids: set[int],
) -> None:
    """Reject token ids that cannot be indexed in the logits list."""
    for token_id in token_ids:
        if token_id < 0 or token_id >= len(logits):
            raise SelectionError(f"Token id out of logits range: {token_id}")
