"""Terminal-only output helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.markup import escape

console = Console()
error_console = Console(stderr=True)

UNKNOWN_MESSAGES = [
    "🤔 I don't know this yet. Teach me something new! 🔥",
    "🧠 Hmm, I’m not sure. Improve me — I love learning ✨",
    "📚 I don’t have this answer yet. Add an example and make me smarter!",
    "🔥 Unknown for now, but not forever. Help me learn!",
]


def print_prompt_start(index: int, total: int, prompt: str) -> None:
    """Print progress for one prompt."""
    console.print(f"[{index}/{total}] Processing: {prompt[:60]}")


def print_function_call(name: str, parameters: dict[str, Any]) -> None:
    """Print the selected function call."""
    args_json = json.dumps(parameters, ensure_ascii=False, sort_keys=True)
    # Use markup=False to avoid issues with special characters
    # in function names or arguments
    console.print(f"  -> {name}({args_json})", markup=False)


def print_unknown_prompt(prompt: str, reason: Exception) -> None:
    """Print a playful terminal-only message for unsupported prompts."""
    error_console.print(f"[unknown] {prompt}")
    error_console.print(random.choice(UNKNOWN_MESSAGES))
    error_console.print(f"Reason: {reason}")


def print_output_saved(path: Path, count: int) -> None:
    """Print output file summary."""
    console.print(f"Saved {count} function calls to {path}")


def print_function_scoring(scored_functions: list[tuple[float, Any]]) -> None:
    """Print a table showing how the LLM scored function candidates."""
    table = Table(
        show_header=True,
        header_style="bold yellow",
        title="[Function Routing]",
    )
    table.add_column("Candidate Name", style="cyan")
    table.add_column("Log-Probability", justify="right", style="green")

    for score, function in scored_functions:
        if score == scored_functions[0][0]:
            table.add_row(f"⭐ {function.name}", f"{score:.4f}", style="bold")
        else:
            table.add_row(str(function.name), f"{score:.4f}", style="dim")

    console.print(table)


def print_candidate_scoring(
    parameter_name: str,
    scored_candidates: list[tuple[float, Any]],
) -> None:
    """Print a table showing how the LLM scored argument candidates."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column(f"Argument LLM Scoring: {parameter_name}", style="cyan")
    table.add_column("Log-Prob", justify="right", style="green")

    for score, candidate in scored_candidates:
        # Escape candidate string to prevent Rich markup issues
        safe_candidate = escape(str(candidate))

        if score == scored_candidates[0][0]:
            table.add_row(f"⭐ {safe_candidate}", f"{score:.4f}", style="bold")
        else:
            table.add_row(safe_candidate, f"{score:.4f}", style="dim")

    console.print(table)
