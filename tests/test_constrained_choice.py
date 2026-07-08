from typing import Any

import pytest

from src.constrained_choice import choose_constrained_candidate
from src.errors import SelectionError


class FakeModel:
    """Fake model with deterministic tokenization and logits."""

    def __init__(self) -> None:
        self.tokens_by_text = {
            " alpha": [1],
            " beta": [2],
            " empty": [],
            " shared_alpha": [10, 1],
            " shared_beta": [10, 2],
        }
        self.logits_by_context = {
            (0,): [0.0, 1.0, 4.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            (0, 10): [
                0.0,
                1.0,
                5.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        }

    def encode(self, text: str) -> list[list[int]]:
        """Return known token ids for candidate text."""
        return [self.tokens_by_text.get(text, [99])]

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return logits for the current fake decoding context."""
        return self.logits_by_context.get(
            tuple(input_ids),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )


def test_chooses_highest_logit_candidate_among_allowed_tokens() -> None:
    """Only candidate next tokens are considered during scoring."""
    score, value = choose_constrained_candidate(
        model=FakeModel(),
        context_ids=[0],
        candidates=[
            (" alpha", "first"),
            (" beta", "second"),
        ],
    )

    assert score > 0.0 - 1.0
    assert value == "second"


def test_rejects_empty_candidate_list() -> None:
    """An empty candidate list cannot be decoded."""
    with pytest.raises(SelectionError):
        choose_constrained_candidate(FakeModel(), [0], [])


def test_rejects_candidates_that_encode_to_empty_token_lists() -> None:
    """Every candidate must have at least one token to score."""
    with pytest.raises(SelectionError):
        choose_constrained_candidate(
            model=FakeModel(),
            context_ids=[0],
            candidates=[(" empty", "value")],
        )


def test_handles_candidates_with_shared_prefix() -> None:
    """Shared prefix tokens are scored with constrained continuations."""
    _, value = choose_constrained_candidate(
        model=FakeModel(),
        context_ids=[0],
        candidates=[
            (" shared_alpha", "first"),
            (" shared_beta", "second"),
        ],
    )

    assert value == "second"


def test_returns_original_candidate_value() -> None:
    """The helper returns the supplied value, not just candidate text."""
    original_value: dict[str, Any] = {"name": "beta", "enabled": True}

    _, value = choose_constrained_candidate(
        model=FakeModel(),
        context_ids=[0],
        candidates=[
            (" alpha", {"name": "alpha"}),
            (" beta", original_value),
        ],
    )

    assert value is original_value
