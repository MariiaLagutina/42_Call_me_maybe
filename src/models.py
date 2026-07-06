from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ParameterType = Literal[
    "string",
    "number",
    "integer",
    "boolean",
    "object",
]


class FunctionParameter(BaseModel):
    """Definition of a single function parameter."""

    model_config = ConfigDict(extra="forbid")

    type: ParameterType
    description: str | None = None


class FunctionDefinition(BaseModel):
    """Definition of a callable function."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, FunctionParameter]
    returns: dict[str, Any] | None = None


class FunctionCallResult(BaseModel):
    """Final output item written to function_calling_results.json."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    name: str = Field(min_length=1)
    parameters: dict[str, Any]
