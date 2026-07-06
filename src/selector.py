"""Function selection using LLM token scores."""

import math
from typing import Any

from src.errors import SelectionError
from src.models import FunctionDefinition
from src.prompt import build_selection_prompt
from src.tokenizer_utils import encode_to_ids
from src.render import print_function_scoring


MAX_SCORED_TOKENS = 2


class FunctionSelector:
    """Selects the best function by scoring candidate function names."""

    def __init__(self, model: Any, min_confidence_gap: float) -> None:
        """Initialize selector.

        Args:
            model: Small_LLM_Model or compatible fake model.
            min_confidence_gap: Minimum score gap between best and second best.
        """
        self.model = model
        self.min_confidence_gap = min_confidence_gap

    def select(
        self,
        user_prompt: str,
        functions: list[FunctionDefinition],
    ) -> FunctionDefinition:
        """Select the most likely function for a user prompt.

        Args:
            user_prompt: Natural-language request.
            functions: Available function definitions.

        Returns:
            Selected function definition.

        Raises:
            SelectionError: If no functions are available or selection is
                uncertain.
        """
        if not functions:
            raise SelectionError("No function definitions available")

        prompt = build_selection_prompt(user_prompt, functions)
        context_ids = encode_to_ids(self.model, prompt)

        encoded_candidates = [
            (function, encode_to_ids(self.model, " " + function.name))
            for function in functions
        ]

        common_prefix = _find_common_prefix(
            [candidate_ids for _, candidate_ids in encoded_candidates]
        )

        scored_functions = [
            (
                self._score_candidate_ids(
                    context_ids=context_ids,
                    common_prefix=common_prefix,
                    candidate_ids=candidate_ids,
                ),
                function,
            )
            for function, candidate_ids in encoded_candidates
        ]

        scored_functions.sort(key=lambda item: item[0], reverse=True)

        # Print scoring results in a table for better visibility
        print_function_scoring(scored_functions)

        best_score, best_function = scored_functions[0]

        if len(scored_functions) == 1:
            return best_function

        second_score = scored_functions[1][0]
        gap = best_score - second_score

        if gap < self.min_confidence_gap:
            raise SelectionError(
                "Function selection is uncertain: "
                f"best={best_function.name}, gap={gap:.4f}"
            )

        return best_function

    def _score_candidate_ids(
        self,
        context_ids: list[int],
        common_prefix: list[int],
        candidate_ids: list[int],
    ) -> float:
        """Score the discriminative part of a candidate function name.

        Common prefix tokens are appended to the context but not scored,
        because they are identical for all candidate function names.

        Args:
            context_ids: Encoded selection prompt.
            common_prefix: Shared token prefix between all candidate names.
            candidate_ids: Encoded candidate function name.

        Returns:
            Sum of log-probabilities for the first discriminative tokens.
        """
        if len(candidate_ids) < len(common_prefix):
            raise SelectionError("Candidate shorter than common prefix")

        current_ids = context_ids + common_prefix
        remaining_ids = candidate_ids[len(common_prefix):]

        if not remaining_ids:
            raise SelectionError("Candidate has no discriminative tokens")

        score = 0.0

        for token_id in remaining_ids[:MAX_SCORED_TOKENS]:
            logits = self.model.get_logits_from_input_ids(current_ids)
            score += _log_probability(logits, token_id)
            current_ids.append(token_id)

        return score


def _find_common_prefix(candidates: list[list[int]]) -> list[int]:
    """Find common token prefix for all candidate token lists.

    Args:
        candidates: Candidate token id lists.

    Returns:
        Shared prefix token ids.
    """
    if not candidates:
        return []

    prefix = candidates[0].copy()

    for candidate in candidates[1:]:
        max_len = min(len(prefix), len(candidate))
        index = 0

        while index < max_len and prefix[index] == candidate[index]:
            index += 1

        prefix = prefix[:index]

        if not prefix:
            break

    return prefix


def _log_probability(logits: list[float], token_id: int) -> float:
    """Compute log-probability of one token from raw logits.

    Args:
        logits: Raw logits for next-token prediction.
        token_id: Token id to score.

    Returns:
        Log-probability of the token.
    """
    if token_id < 0 or token_id >= len(logits):
        raise SelectionError(f"Token id out of logits range: {token_id}")

    max_logit = max(logits)
    log_sum_exp = max_logit + math.log(
        sum(math.exp(logit - max_logit) for logit in logits)
    )
    return logits[token_id] - log_sum_exp
