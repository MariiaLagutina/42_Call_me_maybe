from typing import Any

import pytest

from src.decoder import ArgumentDecoder
from src.errors import DecodingError
from src.models import FunctionDefinition


class FakeScorer:
    """Deterministic scorer used to test decoder candidate construction."""

    def __init__(self, preferred: Any | None = None) -> None:
        """Optionally choose a matching candidate instead of the first one."""
        self.preferred = preferred
        self.calls: list[dict[str, Any]] = []

    def choose(
        self,
        user_prompt: str,
        function_name: str,
        parameter_name: str,
        description: str,
        candidates: list[Any],
    ) -> Any:
        """Choose a deterministic candidate from the supplied list only."""
        self.calls.append(
            {
                "user_prompt": user_prompt,
                "function_name": function_name,
                "parameter_name": parameter_name,
                "description": description,
                "candidates": candidates.copy(),
            }
        )
        if self.preferred in candidates:
            return self.preferred
        return candidates[0]


def make_function(parameters: dict[str, dict[str, str]]) -> FunctionDefinition:
    """Build a FunctionDefinition for decoder tests."""
    return FunctionDefinition.model_validate(
        {
            "name": "fn_test",
            "description": "Test function.",
            "parameters": parameters,
            "returns": {"type": "string"},
        }
    )


def test_decode_number_arguments() -> None:
    """Number parameters are chosen from generated numeric candidates."""
    function = make_function(
        {
            "a": {"type": "number"},
            "b": {"type": "number"},
        }
    )
    scorer = FakeScorer()

    assert ArgumentDecoder(scorer).decode_arguments(
        "Add 2 and 3",
        function,
    ) == {
        "a": 2,
        "b": 3,
    }
    assert scorer.calls[0]["candidates"] == [2, 3]


def test_decode_integer_argument() -> None:
    """Integer parameters are cast to int after candidate scoring."""
    function = make_function({"n": {"type": "integer"}})
    scorer = FakeScorer()

    assert ArgumentDecoder(scorer).decode_arguments(
        "Factorial of 5",
        function,
    ) == {
        "n": 5,
    }


def test_decode_string_from_quotes() -> None:
    """String parameters include quoted literals as candidates."""
    function = make_function({"s": {"type": "string"}})
    scorer = FakeScorer()

    assert ArgumentDecoder(scorer).decode_arguments(
        "Reverse 'hello'",
        function,
    ) == {
        "s": "hello",
    }
    assert scorer.calls[0]["candidates"][0] == "hello"


def test_decode_string_from_last_word_fallback() -> None:
    """Simple unquoted words can become string candidates."""
    function = make_function({"name": {"type": "string"}})
    scorer = FakeScorer()

    assert ArgumentDecoder(scorer).decode_arguments(
        "Greet Alice",
        function,
    ) == {
        "name": "Alice",
    }


def test_decode_boolean_argument() -> None:
    """Boolean parameters are chosen from boolean candidates."""
    function = make_function({"enabled": {"type": "boolean"}})
    scorer = FakeScorer()

    assert ArgumentDecoder(scorer).decode_arguments(
        "Set enabled to true",
        function,
    ) == {
        "enabled": True,
    }


def test_decode_file_path_string_candidate() -> None:
    """File paths are neutral string candidates, not hardcoded parameters."""
    function = make_function({"path": {"type": "string"}})
    scorer = FakeScorer(preferred="/tmp/data.txt")

    assert ArgumentDecoder(scorer).decode_arguments(
        "Read /tmp/data.txt now",
        function,
    ) == {"path": "/tmp/data.txt"}


def test_decode_regex_style_candidate_selection() -> None:
    """Regex-style hints are candidate strings for the scorer to choose."""
    function = make_function(
        {
            "pattern": {"type": "string"},
        }
    )
    scorer = FakeScorer(preferred=r"\d+")

    assert ArgumentDecoder(scorer).decode_arguments(
        "Replace every number with x",
        function,
    ) == {"pattern": r"\d+"}
    assert r"\d+" in scorer.calls[0]["candidates"]


def test_decode_object_returns_empty_object() -> None:
    """Object parameters follow the current empty-object behavior."""
    function = make_function({"payload": {"type": "object"}})
    scorer = FakeScorer()

    assert ArgumentDecoder(scorer).decode_arguments(
        "Send payload",
        function,
    ) == {
        "payload": {},
    }
    assert scorer.calls == []


def test_decode_missing_number_raises_error() -> None:
    """Missing required values are reported as decoding errors."""
    function = make_function({"a": {"type": "number"}})

    with pytest.raises(DecodingError):
        ArgumentDecoder(FakeScorer()).decode_arguments(
            "No number here",
            function,
        )
