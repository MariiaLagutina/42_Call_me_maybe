import pytest

from src.errors import SelectionError
from src.models import FunctionDefinition
from src.selector import (
    FunctionSelector,
    _find_common_prefix,
    _log_probability,
)


class FakeModel:
    """Small fake model with deterministic tokenization and logits."""

    def __init__(
        self,
        add_score: float = 5.0,
        reverse_score: float = 1.0,
    ) -> None:
        """Store deterministic logits for candidate scoring."""
        self.add_score = add_score
        self.reverse_score = reverse_score

    def encode(self, text: str) -> list[list[int]]:
        """Return token ids for the selection prompt or function names."""
        if text.startswith("You are a function-calling router"):
            return [[0]]
        if text == " fn_add_numbers":
            return [[10, 2]]
        if text == " fn_reverse_string":
            return [[10, 3]]
        return [[99]]

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return logits where token 2 and 3 represent candidate names."""
        return [0.0, 0.0, self.add_score, self.reverse_score]


def make_function(name: str) -> FunctionDefinition:
    """Build a minimal function definition."""
    return FunctionDefinition.model_validate(
        {
            "name": name,
            "description": f"Description for {name}.",
            "parameters": {},
            "returns": {"type": "string"},
        }
    )


def test_find_common_prefix() -> None:
    """Common prefix helper returns only shared leading token ids."""
    assert _find_common_prefix([[1, 2, 3], [1, 2, 4], [1, 2]]) == [1, 2]
    assert _find_common_prefix([[1, 2], [3, 4]]) == []


def test_log_probability_prefers_higher_logit() -> None:
    """The token with a higher logit has higher log-probability."""
    logits = [0.0, 3.0]

    assert _log_probability(logits, 1) > _log_probability(logits, 0)


def test_log_probability_rejects_bad_token_id() -> None:
    """Out-of-range token ids become selection errors."""
    with pytest.raises(SelectionError):
        _log_probability([0.0, 1.0], 10)


def test_selector_chooses_highest_scoring_function() -> None:
    """Selector compares valid function names with model scores."""
    functions = [
        make_function("fn_add_numbers"),
        make_function("fn_reverse_string"),
    ]
    selector = FunctionSelector(FakeModel(), min_confidence_gap=0.01)

    selected = selector.select("What is the sum of 2 and 3?", functions)

    assert selected.name == "fn_add_numbers"


def test_selector_rejects_uncertain_selection() -> None:
    """Equal candidate scores are rejected when confidence gap is required."""
    functions = [
        make_function("fn_add_numbers"),
        make_function("fn_reverse_string"),
    ]
    selector = FunctionSelector(
        FakeModel(add_score=1.0, reverse_score=1.0),
        min_confidence_gap=0.01,
    )

    with pytest.raises(SelectionError):
        selector.select("Ambiguous prompt", functions)
