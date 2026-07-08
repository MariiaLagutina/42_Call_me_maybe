"""Vocabulary helpers used by token-level constraints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TokenizerVocabError(RuntimeError):
    """Raised when tokenizer vocabulary cannot be loaded."""


class TokenizerVocab:
    """Map tokenizer ids to decoded text pieces."""

    def __init__(self, token_to_id: dict[str, int], model: Any) -> None:
        if not token_to_id:
            raise TokenizerVocabError("Tokenizer vocabulary is empty")

        self._token_to_id = token_to_id
        self._id_to_token = {
            int(token_id): token
            for token, token_id in token_to_id.items()
        }
        self._model = model

    @classmethod
    def from_model(cls, model: Any) -> "TokenizerVocab":
        """Build vocabulary from the files exposed by llm_sdk."""
        paths: list[tuple[Path, bool]] = []

        if hasattr(model, "get_path_to_tokenizer_file"):
            paths.append((Path(model.get_path_to_tokenizer_file()), True))

        if hasattr(model, "get_path_to_vocab_file"):
            paths.append((Path(model.get_path_to_vocab_file()), False))

        errors: list[str] = []

        for path, is_tokenizer_file in paths:
            try:
                return cls(_read_token_map(path, is_tokenizer_file), model)
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"{path}: {exc}")

        detail = "; ".join(errors) if errors else "no tokenizer/vocab path"
        raise TokenizerVocabError(
            f"Cannot load tokenizer vocabulary: {detail}"
        )

    def id_to_text(self, token_id: int) -> str:
        """Return decoded text for one token id."""
        try:
            return str(self._model.decode([token_id]))
        except Exception:
            return self._id_to_token.get(token_id, "")

    def id_to_text_map(self) -> dict[int, str]:
        """Return decoded text for every known token id."""
        return {
            token_id: self.id_to_text(token_id)
            for token_id in sorted(self._id_to_token)
        }


def _read_token_map(path: Path, is_tokenizer_file: bool) -> dict[str, int]:
    """Read token -> id mapping from tokenizer.json or vocab.json."""
    data = json.loads(path.read_text(encoding="utf-8"))

    if not is_tokenizer_file:
        return _normalize_token_map(data)

    if not isinstance(data, dict):
        raise ValueError("tokenizer file must contain a JSON object")

    model_data = data.get("model")
    if not isinstance(model_data, dict):
        raise ValueError("tokenizer file missing model object")

    vocab_data = model_data.get("vocab")
    return _normalize_token_map(vocab_data)


def _normalize_token_map(raw: object) -> dict[str, int]:
    """Validate and normalize a token mapping."""
    if not isinstance(raw, dict):
        raise ValueError("vocabulary must be a JSON object")

    out: dict[str, int] = {}
    for token, token_id in raw.items():
        if not isinstance(token, str) or not isinstance(token_id, int):
            raise ValueError(
                "vocabulary must map string tokens to integer ids"
            )
        out[token] = token_id

    if not out:
        raise ValueError("vocabulary is empty")

    return out
