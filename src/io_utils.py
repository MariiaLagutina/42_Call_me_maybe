import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.errors import (
    FileReadError,
    FileWriteError,
    JsonFormatError,
    SchemaValidationError,
)
from src.models import FunctionCallResult, FunctionDefinition


def read_json_file(path: Path) -> Any:
    """Read and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data.

    Raises:
        FileReadError: If the file is missing or cannot be read.
        JsonFormatError: If the file is empty or contains invalid JSON.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileReadError(f"File not found: {path}") from e
    except OSError as e:
        raise FileReadError(f"Cannot read file {path}: {e}") from e

    if not content.strip():
        raise JsonFormatError(f"File is empty: {path}")

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        message = (
            f"Invalid JSON in {path}: line {e.lineno}, "
            f"column {e.colno}: {e.msg}"
        )
        raise JsonFormatError(message) from e


def load_function_definitions(path: Path) -> list[FunctionDefinition]:
    """Load and validate function definitions.

    Args:
        path: Path to functions_definition.json.

    Returns:
        List of validated function definitions.

    Raises:
        SchemaValidationError: If the JSON structure is invalid.
    """
    data = read_json_file(path)

    if not isinstance(data, list):
        raise SchemaValidationError(
            f"Function definitions file must contain a JSON array: {path}"
        )

    try:
        return [FunctionDefinition.model_validate(item) for item in data]
    except ValidationError as e:
        raise SchemaValidationError(
            f"Invalid function definition schema in {path}: {e}"
        ) from e


def load_test_prompts(path: Path) -> list[str]:
    """Load natural-language prompts from JSON file.

    Args:
        path: Path to function_calling_tests.json.

    Returns:
        List of prompts.

    Raises:
        SchemaValidationError: If the file does not contain valid prompts.
    """
    data = read_json_file(path)

    if not isinstance(data, list):
        raise SchemaValidationError(
            f"Test prompts file must contain a JSON array: {path}"
        )

    prompts: list[str] = []

    for index, item in enumerate(data):
        if isinstance(item, str):
            prompt = item
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            prompt = item["prompt"]
        else:
            raise SchemaValidationError(
                "Prompt at index "
                f"{index} must be a string or an object with a string "
                "'prompt' field"
            )

        if not prompt.strip():
            raise SchemaValidationError(f"Prompt at index {index} is empty")

        prompts.append(prompt)

    return prompts


def write_results(path: Path, results: list[FunctionCallResult]) -> None:
    """Write function call results to a JSON file.

    Args:
        path: Output JSON path.
        results: Validated function call results.

    Raises:
        FileWriteError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [result.model_dump() for result in results]

    try:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        raise FileWriteError(
            f"Cannot write output file {path}: {e}"
        ) from e

    # Safety check: make sure the written file is parseable JSON.
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise FileWriteError(
            f"Output file was written but is not valid JSON: {path}"
        ) from e
