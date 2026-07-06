import pytest
from pydantic import ValidationError

from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    FunctionParameter,
)


def test_function_parameter_accepts_supported_type() -> None:
    """FunctionParameter accepts one of the supported schema types."""
    parameter = FunctionParameter(type="string", description="User name")

    assert parameter.type == "string"
    assert parameter.description == "User name"


def test_function_parameter_rejects_unknown_type() -> None:
    """Unsupported parameter types are rejected by Pydantic."""
    with pytest.raises(ValidationError):
        FunctionParameter(type="array")  # type: ignore[arg-type]


def test_function_definition_rejects_extra_fields() -> None:
    """FunctionDefinition must not silently accept unknown fields."""
    with pytest.raises(ValidationError):
        FunctionDefinition.model_validate(
            {
                "name": "fn_add_numbers",
                "description": "Add two numbers.",
                "parameters": {},
                "returns": {"type": "number"},
                "extra": "not allowed",
            }
        )


def test_function_call_result_uses_expected_output_keys() -> None:
    """Output model keeps the project output schema stable."""
    result = FunctionCallResult(
        prompt="What is the sum of 2 and 3?",
        name="fn_add_numbers",
        parameters={"a": 2.0, "b": 3.0},
    )

    assert result.model_dump() == {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {"a": 2.0, "b": 3.0},
    }
