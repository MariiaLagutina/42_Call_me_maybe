import math

from src.logit_utils import apply_mask, argmax, cumulative_sequence_logprob


def test_apply_mask_keeps_only_allowed_token_ids() -> None:
    """Disallowed logits are hidden from argmax."""
    masked = apply_mask([1.0, 9.0, 3.0], {0, 2})

    assert masked[0] == 1.0
    assert masked[1] == -math.inf
    assert masked[2] == 3.0
    assert argmax(masked) == 2


def test_cumulative_sequence_logprob_scores_all_tokens() -> None:
    """Multi-token candidates are scored token by token."""
    calls: list[list[int]] = []

    def fake_logits(input_ids: list[int]) -> list[float]:
        calls.append(input_ids.copy())
        return [0.0, 5.0, 4.0]

    score = cumulative_sequence_logprob(fake_logits, [0], [1, 2])

    assert score < 0
    assert calls == [[0], [0, 1]]
