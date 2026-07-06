from pathlib import Path

from rich.console import Console

from src import render


def use_test_console(monkeypatch, tmp_path: Path) -> Path:
    """Patch render.console to capture output in a temporary file."""
    output_path = tmp_path / "terminal.txt"
    stream = output_path.open("w", encoding="utf-8")
    monkeypatch.setattr(
        render,
        "console",
        Console(file=stream, force_terminal=False, width=100),
    )
    return output_path


def test_print_function_call_writes_terminal_output_without_mutating_data(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Function-call rendering is terminal-only and leaves data intact."""
    output_path = use_test_console(monkeypatch, tmp_path)
    parameters = {"pattern": "[abc]", "count": 2}

    render.print_function_call("fn_match", parameters)
    render.console.file.close()

    assert "fn_match" in output_path.read_text(encoding="utf-8")
    assert parameters == {"pattern": "[abc]", "count": 2}


def test_print_candidate_scoring_writes_some_table_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Rich table details are not asserted too tightly."""
    output_path = use_test_console(monkeypatch, tmp_path)

    render.print_candidate_scoring("value", [(1.0, "alpha"), (0.1, "beta")])
    render.console.file.close()

    output = output_path.read_text(encoding="utf-8")
    assert "value" in output
    assert "alpha" in output
