import math
from typing import Any

from src.errors import SelectionError
from src.tokenizer_utils import encode_to_ids
from src.render import print_candidate_scoring


class CandidateScorer:
    """Scores a list of possible argument values using model logits."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def choose(
        self,
        user_prompt: str,
        function_name: str,
        parameter_name: str,
        description: str,
        candidates: list[Any],
    ) -> Any:
        """Score candidates and return the most likely one."""
        if not candidates:
            raise SelectionError(
                f"No candidates available for {parameter_name}"
            )

        # If candidates list has only one item, return it directly
        if len(candidates) == 1:
            return candidates[0]

        # Build context for scoring candidates
        context = (
            f"User request: {user_prompt}\n"
            f"Function: {function_name}\n"
            f"Parameter to extract: {parameter_name} ({description})\n"
            "Exact value:"
        )
        context_ids = encode_to_ids(self.model, context)

        scored_candidates = []
        for candidate in candidates:
            # Model works with text, so we convert the candidate to a string
            # Add a space because tokens often start with a leading space.
            candidate_str = " " + str(candidate)
            candidate_ids = encode_to_ids(self.model, candidate_str)

            score = self._score_tokens(context_ids, candidate_ids)
            scored_candidates.append((score, candidate))

        # Sort by descending log-probability
        scored_candidates.sort(key=lambda item: item[0], reverse=True)

        # Print scoring results in a table for better visibility
        print_candidate_scoring(parameter_name, scored_candidates)

        return scored_candidates[0][1]

    def _score_tokens(
        self,
        context_ids: list[int],
        candidate_ids: list[int],
    ) -> float:
        """Sum log probabilities for the candidate tokens."""
        if not candidate_ids:
            return -float("inf")

        score = 0.0
        current_ids = context_ids.copy()

        # Score the first 2-3 tokens of the candidate for speed and accuracy
        # (as in your function selector)
        for token_id in candidate_ids[:3]:
            logits = self.model.get_logits_from_input_ids(current_ids)
            score += self._log_probability(logits, token_id)
            current_ids.append(token_id)

        return score

    def _log_probability(self, logits: list[float], token_id: int) -> float:
        """Compute log-probability of one token from raw logits."""
        if token_id < 0 or token_id >= len(logits):
            raise SelectionError(f"Token id out of logits range: {token_id}")

        max_logit = max(logits)
        log_sum_exp = max_logit + math.log(
            sum(math.exp(logit - max_logit) for logit in logits)
        )
        return logits[token_id] - log_sum_exp
