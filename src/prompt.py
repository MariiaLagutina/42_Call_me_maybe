from src.models import FunctionDefinition, FunctionParameter


def build_selection_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Build a deterministic prompt for function selection.
    The model is expected to continue after "Correct function:" with the
    most appropriate function name.
    """
    function_lines = [
        f"- {function.name}: {function.description}"
        for function in functions
    ]
    functions_block = "\n".join(function_lines)

    return (
        "You are a function-calling router.\n"
        "Choose the single best function for the user request.\n\n"
        "Available functions:\n"
        f"{functions_block}\n\n"
        f"User request:\n{user_prompt}\n\n"
        "Correct function:"
    )


def build_argument_prompt(
    user_prompt: str,
    function_name: str,
    parameter_name: str,
    parameter: FunctionParameter,
) -> str:
    """Build a deterministic prompt for decoding one argument value.
    """
    description = parameter.description or "No description provided."

    return (
        "Extract one argument value for a function call.\n\n"
        f"User request:\n{user_prompt}\n\n"
        f"Function:\n{function_name}\n\n"
        f"Argument name:\n{parameter_name}\n\n"
        f"Argument type:\n{parameter.type}\n\n"
        f"Argument description:\n{description}\n\n"
        "JSON value:"
    )
