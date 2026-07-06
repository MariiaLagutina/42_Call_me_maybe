from typing import Any

import pytest

from src.candidate_scorer import CandidateScorer
from src.errors import SelectionError


class FakeModel:
    """Fake model exposing only the methods used by CandidateScorer."""

    def __init__(self, preferred_token: int) -> None:
        """Store the token that should receive the highest fake score."""
        self.preferred_token = preferred_token

    def encode(self, text: str) -> list[list[int]]:
        """Map known candidate strings to one-token ids."""
        token_by_text = {
            "User request": 0,
            " alpha": 1,
            " beta": 2,
            " gamma": 3,
        }
        for prefix, token_id in token_by_text.items():
            if text.startswith(prefix):
                return [[token_id]]
        return [[0]]

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return logits that prefer one configured token id."""
        logits = [0.0, 0.0, 0.0, 0.0]
        logits[self.preferred_token] = 5.0
        return logits


def disable_render(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep scoring tests focused on return values, not terminal output."""
    monkeypatch.setattr(
        "src.candidate_scorer.print_candidate_scoring",
        lambda parameter_name, scored_candidates: None,
    )


def choose(
    scorer: CandidateScorer,
    candidates: list[Any],
) -> Any:
    """Call CandidateScorer with stable prompt metadata."""
    return scorer.choose(
        user_prompt="Choose a value",
        function_name="fn_test",
        parameter_name="value",
        description="A test value.",
        candidates=candidates,
    )


def test_choose_returns_highest_log_probability_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scorer chooses the candidate with the highest fake logit."""
    disable_render(monkeypatch)
    scorer = CandidateScorer(FakeModel(preferred_token=2))

    assert choose(scorer, ["alpha", "beta", "gamma"]) == "beta"


def test_choose_rejects_empty_candidate_list() -> None:
    """Empty candidate lists become project selection errors."""
    scorer = CandidateScorer(FakeModel(preferred_token=1))

    with pytest.raises(SelectionError):
        choose(scorer, [])


def test_choose_only_returns_supplied_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scoring is constrained to candidates and never free-form text."""
    disable_render(monkeypatch)
    scorer = CandidateScorer(FakeModel(preferred_token=3))
    candidates = ["alpha", "beta"]

    assert choose(scorer, candidates) in candidates
