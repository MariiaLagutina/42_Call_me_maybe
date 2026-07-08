"""Build allowed token sets for constrained value generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.tokenizer_utils import encode_to_ids
from src.tokenizer_vocab import TokenizerVocab


@dataclass(frozen=True)
class TokenConstraints:
    """Precomputed token ids used by constrained decoding."""

    piece_by_id: dict[int, str]
    quote_ids: list[int]
    true_ids: list[int]
    false_ids: list[int]
    safe_string_ids: set[int]
    number_ids: set[int]

    @classmethod
    def from_model(cls, model: Any) -> "TokenConstraints":
        """Build token constraints from the model tokenizer vocabulary."""
        vocab = TokenizerVocab.from_model(model)
        piece_by_id = vocab.id_to_text_map()
        quote_ids = encode_to_ids(model, '"')
        true_ids = encode_to_ids(model, "true")
        false_ids = encode_to_ids(model, "false")

        return cls(
            piece_by_id=piece_by_id,
            quote_ids=quote_ids,
            true_ids=true_ids,
            false_ids=false_ids,
            safe_string_ids=_build_safe_string_ids(piece_by_id, quote_ids),
            number_ids=_build_number_ids(piece_by_id),
        )


def normalize_piece(piece: str) -> str:
    """Normalize common BPE/SentencePiece surface markers."""
    return (
        piece.replace("Ġ", " ")
        .replace("Ċ", "")
        .replace("▁", " ")
        .replace("Äł", "")
    )


def _build_safe_string_ids(
    piece_by_id: dict[int, str],
    quote_ids: list[int],
) -> set[int]:
    """Allow string-body tokens plus the quote token that can close strings."""
    forbidden = {'"', "\n", "\r"}
    token_ids = {
        token_id
        for token_id, piece in piece_by_id.items()
        if piece and not any(char in piece for char in forbidden)
    }
    token_ids.update(quote_ids)
    return token_ids


def _build_number_ids(piece_by_id: dict[int, str]) -> set[int]:
    """Allow only token pieces that can be part of a JSON number."""
    valid_chars = set("0123456789.-")
    token_ids: set[int] = set()

    for token_id, piece in piece_by_id.items():
        stripped = normalize_piece(piece).strip()
        if stripped and all(char in valid_chars for char in stripped):
            token_ids.add(token_id)

    return token_ids
