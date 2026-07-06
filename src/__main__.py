import argparse
import sys
from pathlib import Path

from src.config import (
    DEFAULT_FUNCTIONS_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_PROMPTS_PATH,
)
from src.errors import CallMeMaybeError
from src.pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert natural-language prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS_PATH,
        help="Path to functions_definition.json",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PROMPTS_PATH,
        help="Path to function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name used by llm_sdk",
    )
    return parser


def main() -> int:
    """Run the CLI application."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        pipeline = Pipeline(
            functions_path=args.functions_definition,
            prompts_path=args.input,
            output_path=args.output,
            model_name=args.model_name,
        )
        pipeline.run()
    except CallMeMaybeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
