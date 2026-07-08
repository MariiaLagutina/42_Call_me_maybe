from typing import Any

from src.errors import SelectionError
from src.logit_utils import cumulative_sequence_logprob
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

        context = (
            f"User request: {user_prompt}\n"
            f"Function: {function_name}\n"
            f"Parameter to extract: {parameter_name} ({description})\n"
            "Exact value:"
        )
        context_ids = encode_to_ids(self.model, context)

        scored_candidates = []
        for candidate in candidates:
            candidate_str = " " + str(candidate)
            candidate_ids = encode_to_ids(self.model, candidate_str)

            score = self._score_tokens(context_ids, candidate_ids)
            scored_candidates.append((score, candidate))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)

        print_candidate_scoring(parameter_name, scored_candidates)

        return scored_candidates[0][1]

    def _score_tokens(
        self,
        context_ids: list[int],
        candidate_ids: list[int],
    ) -> float:
        """Sum log probabilities for the candidate tokens."""
        return cumulative_sequence_logprob(
            self.model.get_logits_from_input_ids,
            context_ids,
            candidate_ids,
        )
