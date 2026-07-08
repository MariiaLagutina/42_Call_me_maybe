"""Fallback token-masked value generation for function arguments."""

from __future__ import annotations

from typing import Any

from src.logit_utils import apply_mask, argmax, cumulative_sequence_logprob
from src.token_constraints import TokenConstraints, normalize_piece
from src.tokenizer_utils import encode_to_ids


class ConstrainedValueDecoder:
    """Generate simple JSON values with token masks."""

    def __init__(
        self,
        model: Any,
        constraints: TokenConstraints,
        max_string_tokens: int = 30,
        max_number_tokens: int = 12,
    ) -> None:
        self.model = model
        self.constraints = constraints
        self.max_string_tokens = max_string_tokens
        self.max_number_tokens = max_number_tokens

    @classmethod
    def try_from_model(cls, model: Any) -> "ConstrainedValueDecoder | None":
        """Build decoder when tokenizer files are available."""
        try:
            return cls(model, TokenConstraints.from_model(model))
        except Exception:
            return None

    def choose_boolean(self, context: str) -> bool:
        """Choose JSON true/false by scoring only those two literals."""
        context_ids = encode_to_ids(self.model, context)
        get_logits = self.model.get_logits_from_input_ids
        true_score = cumulative_sequence_logprob(
            get_logits,
            context_ids,
            self.constraints.true_ids,
        )
        false_score = cumulative_sequence_logprob(
            get_logits,
            context_ids,
            self.constraints.false_ids,
        )
        return true_score >= false_score

    def generate_number(self, context: str, integer_only: bool) -> int | float:
        """Generate a numeric value using only numeric token ids."""
        current_ids = encode_to_ids(self.model, context)
        value_text = ""

        for _ in range(self.max_number_tokens):
            logits = self.model.get_logits_from_input_ids(current_ids)
            next_id = argmax(apply_mask(logits, self.constraints.number_ids))
            piece = normalize_piece(
                self.constraints.piece_by_id.get(next_id, "")
            ).strip()

            if not piece:
                break

            next_text = value_text + piece
            if not _is_possible_number_prefix(next_text, integer_only):
                break

            value_text = next_text
            current_ids.append(next_id)

            if _is_complete_number(value_text, integer_only):
                break

        if not value_text or not _is_complete_number(value_text, integer_only):
            return 0 if integer_only else 0.0

        return int(value_text) if integer_only else float(value_text)

    def generate_string(self, context: str) -> str:
        """Generate a JSON string body using safe string tokens."""
        current_ids = encode_to_ids(self.model, context + '"')
        quote_ids = set(self.constraints.quote_ids)
        value_parts: list[str] = []

        for _ in range(self.max_string_tokens):
            logits = self.model.get_logits_from_input_ids(current_ids)
            next_id = argmax(
                apply_mask(logits, self.constraints.safe_string_ids)
            )
            current_ids.append(next_id)

            if next_id in quote_ids:
                break

            piece = normalize_piece(
                self.constraints.piece_by_id.get(next_id, "")
            )
            value_parts.append(piece)

        return "".join(value_parts).strip()


def _is_possible_number_prefix(value: str, integer_only: bool) -> bool:
    """Return whether text can still become a valid JSON number."""
    if value in {"-", ".", "-."}:
        return not integer_only or value == "-"
    if integer_only and "." in value:
        return False
    return all(char in "0123456789.-" for char in value)


def _is_complete_number(value: str, integer_only: bool) -> bool:
    """Return whether text is parseable as the requested numeric type."""
    try:
        if integer_only:
            int(value)
        else:
            float(value)
    except ValueError:
        return False
    return True
