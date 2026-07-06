"""Main project pipeline."""

from pathlib import Path

from llm_sdk import Small_LLM_Model

from src.config import MIN_SELECTION_CONFIDENCE
from src.decoder import ArgumentDecoder
from src.io_utils import (
    load_function_definitions,
    load_test_prompts,
    write_results,
)
from src.errors import DecodingError, SelectionError
from src.render import (
    print_function_call,
    print_output_saved,
    print_prompt_start,
    print_unknown_prompt,
)
from src.models import FunctionCallResult
from src.selector import FunctionSelector
from src.candidate_scorer import CandidateScorer


class Pipeline:
    """Orchestrates loading, processing prompts, and writing output."""

    def __init__(
        self,
        functions_path: Path,
        prompts_path: Path,
        output_path: Path,
        model_name: str,
    ) -> None:
        """Initialize the pipeline.

        Args:
            functions_path: Path to function definitions JSON.
            prompts_path: Path to prompts JSON.
            output_path: Path to output JSON.
            model_name: Name of the model used by llm_sdk.
        """
        self.functions_path = functions_path
        self.prompts_path = prompts_path
        self.output_path = output_path
        self.model_name = model_name

    def run(self) -> None:
        """Run the full pipeline."""
        functions = load_function_definitions(self.functions_path)
        prompts = load_test_prompts(self.prompts_path)

        if not functions:
            raise ValueError("No function definitions found")

        model = Small_LLM_Model(self.model_name)
        selector = FunctionSelector(
            model=model,
            min_confidence_gap=MIN_SELECTION_CONFIDENCE,
        )

        scorer = CandidateScorer(model=model)
        decoder = ArgumentDecoder(scorer=scorer)

        results: list[FunctionCallResult] = []

        for index, prompt in enumerate(prompts, start=1):
            print_prompt_start(index, len(prompts), prompt)

            try:
                selected_function = selector.select(prompt, functions)
                args = decoder.decode_arguments(prompt, selected_function)
            except (SelectionError, DecodingError) as exc:
                print_unknown_prompt(prompt, exc)
                continue

            print_function_call(selected_function.name, args)

            results.append(
                FunctionCallResult(
                    prompt=prompt,
                    name=selected_function.name,
                    parameters=args,
                )
            )

        write_results(self.output_path, results)
        print_output_saved(self.output_path, len(results))
