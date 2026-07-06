import json
from pathlib import Path

import pytest

from src.errors import FileReadError, JsonFormatError, SchemaValidationError
from src.io_utils import (
    load_function_definitions,
    load_test_prompts,
    read_json_file,
    write_results,
)
from src.models import FunctionCallResult


def write_json(path: Path, data: object) -> None:
    """Write JSON test data."""
    path.write_text(json.dumps(data), encoding="utf-8")


def test_read_json_file_reports_missing_file(tmp_path: Path) -> None:
    """Missing files are converted to project errors."""
    with pytest.raises(FileReadError):
        read_json_file(tmp_path / "missing.json")


def test_read_json_file_reports_invalid_json(tmp_path: Path) -> None:
    """Malformed JSON is converted to JsonFormatError."""
    path = tmp_path / "bad.json"
    path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(JsonFormatError):
        read_json_file(path)


def test_load_function_definitions_validates_schema(tmp_path: Path) -> None:
    """Function definitions are parsed into Pydantic models."""
    path = tmp_path / "functions_definition.json"
    write_json(
        path,
        [
            {
                "name": "fn_add_numbers",
                "description": "Add two numbers.",
                "parameters": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "returns": {"type": "number"},
            }
        ],
    )

    functions = load_function_definitions(path)

    assert len(functions) == 1
    assert functions[0].name == "fn_add_numbers"
    assert functions[0].parameters["a"].type == "number"


def test_load_function_definitions_rejects_non_array(tmp_path: Path) -> None:
    """Function definitions file must contain a JSON array."""
    path = tmp_path / "functions_definition.json"
    write_json(path, {"name": "fn_add_numbers"})

    with pytest.raises(SchemaValidationError):
        load_function_definitions(path)


def test_load_test_prompts_accepts_strings_and_prompt_objects(
    tmp_path: Path,
) -> None:
    """Prompt loader accepts both supported prompt formats."""
    path = tmp_path / "function_calling_tests.json"
    write_json(
        path,
        [
            "Greet shrek",
            {"prompt": "What is the sum of 2 and 3?"},
        ],
    )

    assert load_test_prompts(path) == [
        "Greet shrek",
        "What is the sum of 2 and 3?",
    ]


def test_load_test_prompts_rejects_empty_prompt(tmp_path: Path) -> None:
    """Empty prompts are rejected before the model is called."""
    path = tmp_path / "function_calling_tests.json"
    write_json(path, [{"prompt": "   "}])

    with pytest.raises(SchemaValidationError):
        load_test_prompts(path)


def test_write_results_creates_parseable_json(tmp_path: Path) -> None:
    """Output writer creates directories and writes parseable JSON."""
    output = tmp_path / "nested" / "function_calling_results.json"
    result = FunctionCallResult(
        prompt="Greet shrek",
        name="fn_greet",
        parameters={"name": "shrek"},
    )

    write_results(output, [result])

    assert json.loads(output.read_text(encoding="utf-8")) == [
        {
            "prompt": "Greet shrek",
            "name": "fn_greet",
            "parameters": {"name": "shrek"},
        }
    ]
