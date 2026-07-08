from __future__ import annotations

from typing import Any

from src.candidate_scorer import CandidateScorer
from src.constrained_value_decoder import ConstrainedValueDecoder
from src.errors import DecodingError
from src.extractor import (
    extract_booleans,
    extract_file_path,
    extract_last_word,
    extract_numbers,
    extract_quoted_strings,
    generate_regex_candidates,
    generate_replacement_candidates,
    extract_word_before_keyword,
    extract_after_colon,
)
from src.models import FunctionDefinition


def _unique_candidates(candidates: list[Any]) -> list[Any]:
    """Return candidates without duplicates, preserving order."""
    unique: list[Any] = []

    for candidate in candidates:
        if candidate is not None and candidate not in unique:
            unique.append(candidate)

    return unique


def _string_candidates_for_parameter(
    parameter_name: str,
    user_prompt: str,
) -> list[str]:
    """Build string candidates according to the parameter role."""
    parameter_name_lower = parameter_name.lower()

    quoted_strings = extract_quoted_strings(user_prompt)
    last_word = extract_last_word(user_prompt)
    file_path = extract_file_path(user_prompt)
    regex_hints = generate_regex_candidates(user_prompt)
    replacement_hints = generate_replacement_candidates(user_prompt)

    candidates: list[str] = []

    if "query" in parameter_name_lower:
        candidates.extend(quoted_strings)

    elif "database" in parameter_name_lower:
        candidates.extend(
            extract_word_before_keyword(user_prompt, "database")
        )

    elif "encoding" in parameter_name_lower:
        candidates.extend(
            extract_word_before_keyword(user_prompt, "encoding")
        )

    elif "template" in parameter_name_lower:
        template = extract_after_colon(user_prompt)
        if template is not None:
            candidates.append(template)

    elif "source" in parameter_name_lower:
        candidates = sorted(quoted_strings, key=len, reverse=True)

    elif "regex" in parameter_name_lower or "pattern" in parameter_name_lower:
        candidates.extend(regex_hints)

        if not candidates and quoted_strings:
            candidates.append(quoted_strings[0])

    elif "replacement" in parameter_name_lower:
        if replacement_hints:
            candidates.extend(replacement_hints)
        else:
            if len(quoted_strings) >= 2:
                candidates.append(quoted_strings[1])
            if last_word is not None:
                candidates.append(last_word)

    elif "path" in parameter_name_lower and file_path is not None:
        candidates.append(file_path)

    elif quoted_strings:
        candidates.extend(quoted_strings)

    elif last_word is not None:
        candidates.append(last_word)

    return _unique_candidates(candidates)


class ArgumentDecoder:
    """Decode function arguments using schema-guided candidate generation."""

    def __init__(
        self,
        scorer: CandidateScorer,
        value_decoder: ConstrainedValueDecoder | None = None,
    ) -> None:
        self.scorer = scorer
        self.value_decoder = value_decoder

    def decode_arguments(
        self,
        user_prompt: str,
        function: FunctionDefinition,
    ) -> dict[str, Any]:
        """Decode all arguments required by a selected function."""
        numbers = extract_numbers(user_prompt)
        booleans = extract_booleans(user_prompt)

        args: dict[str, Any] = {}

        for parameter_name, parameter in function.parameters.items():
            current_candidates: list[Any]

            if parameter.type in {"number", "integer"}:
                current_candidates = numbers

            elif parameter.type == "string":
                current_candidates = _string_candidates_for_parameter(
                    parameter_name,
                    user_prompt,
                )

            elif parameter.type == "boolean":
                current_candidates = booleans

            elif parameter.type == "object":
                args[parameter_name] = {}
                continue

            else:
                raise DecodingError(
                    f"Unsupported parameter type: {parameter.type}"
                )

            if current_candidates:
                best_value = self.scorer.choose(
                    user_prompt=user_prompt,
                    function_name=function.name,
                    parameter_name=parameter_name,
                    description=parameter.description or "No description",
                    candidates=current_candidates,
                )
            else:
                best_value = self._decode_with_token_mask(
                    user_prompt=user_prompt,
                    function_name=function.name,
                    parameter_name=parameter_name,
                    description=parameter.description or "No description",
                    parameter_type=parameter.type,
                )

            if parameter.type in {"number", "integer"}:
                if best_value in numbers:
                    numbers.remove(best_value)

            elif parameter.type == "boolean":
                if best_value in booleans:
                    booleans.remove(best_value)

            if parameter.type == "integer":
                best_value = int(best_value)

            elif parameter.type == "number":
                best_value = float(best_value)

            args[parameter_name] = best_value

        return args

    def _decode_with_token_mask(
        self,
        user_prompt: str,
        function_name: str,
        parameter_name: str,
        description: str,
        parameter_type: str,
    ) -> Any:
        """Fallback to token-masked generation when extraction has no value."""
        if self.value_decoder is None:
            raise DecodingError(
                "No candidate values found in prompt for "
                f"{parameter_name}"
            )

        context = (
            f"User request: {user_prompt}\n"
            f"Function: {function_name}\n"
            f"Parameter: {parameter_name} ({description})\n"
            "JSON value: "
        )

        if parameter_type == "boolean":
            return self.value_decoder.choose_boolean(context)

        if parameter_type in {"number", "integer"}:
            return self.value_decoder.generate_number(
                context,
                integer_only=(parameter_type == "integer"),
            )

        if parameter_type == "string":
            return self.value_decoder.generate_string(context)

        raise DecodingError(
            f"Token-masked fallback does not support {parameter_type}"
        )
