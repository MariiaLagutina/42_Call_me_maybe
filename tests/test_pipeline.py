import importlib
import json
import sys
import types
from pathlib import Path


class FakeSmallLLMModel:
    """Fake llm_sdk.Small_LLM_Model used to avoid loading a real model."""

    def __init__(self, model_name: str) -> None:
        """Store the model name for debugging."""
        self.model_name = model_name

    def encode(self, text: str) -> list[list[int]]:
        """Return deterministic token ids for functions and arguments."""
        if text.startswith("You are a function-calling router"):
            return [[0]]
        if text == " fn_add_numbers":
            return [[10, 2]]
        if text == " fn_reverse_string":
            return [[10, 3]]
        if text == " 2":
            return [[2]]
        if text == " 3":
            return [[4]]
        return [[99]]

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Prefer fn_add_numbers and the first numeric argument."""
        return [0.0, 0.0, 5.0, 1.0, 2.0]


def import_pipeline_with_fake_sdk() -> type:
    """Import Pipeline after installing a fake llm_sdk module."""
    fake_sdk = types.ModuleType("llm_sdk")
    fake_sdk.Small_LLM_Model = FakeSmallLLMModel
    sys.modules["llm_sdk"] = fake_sdk
    sys.modules.pop("src.pipeline", None)

    pipeline_module = importlib.import_module("src.pipeline")
    return pipeline_module.Pipeline


def write_json(path: Path, data: object) -> None:
    """Write JSON test data."""
    path.write_text(json.dumps(data), encoding="utf-8")


def test_pipeline_writes_function_call_results(tmp_path: Path) -> None:
    """The pipeline can process prompts and write the expected JSON output."""
    pipeline_class = import_pipeline_with_fake_sdk()
    functions_path = tmp_path / "functions_definition.json"
    prompts_path = tmp_path / "function_calling_tests.json"
    output_path = tmp_path / "output" / "function_calling_results.json"

    write_json(
        functions_path,
        [
            {
                "name": "fn_add_numbers",
                "description": "Add two numbers.",
                "parameters": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "returns": {"type": "number"},
            },
            {
                "name": "fn_reverse_string",
                "description": "Reverse a string.",
                "parameters": {"s": {"type": "string"}},
                "returns": {"type": "string"},
            },
        ],
    )
    write_json(
        prompts_path,
        [{"prompt": "What is the sum of 2 and 3?"}],
    )

    pipeline = pipeline_class(
        functions_path=functions_path,
        prompts_path=prompts_path,
        output_path=output_path,
        model_name="fake-model",
    )
    pipeline.run()

    results = json.loads(output_path.read_text(encoding="utf-8"))

    assert results == [
        {
            "prompt": "What is the sum of 2 and 3?",
            "name": "fn_add_numbers",
            "parameters": {"a": 2, "b": 3},
        }
    ]
    assert set(results[0]) == {"prompt", "name", "parameters"}
