import json
from pathlib import Path

from src.token_constraints import TokenConstraints


class FakeVocabModel:
    """Model exposing just enough tokenizer API for constraints."""

    def __init__(self, vocab_path: Path) -> None:
        self.vocab_path = vocab_path

    def get_path_to_vocab_file(self) -> str:
        return str(self.vocab_path)

    def encode(self, text: str) -> list[list[int]]:
        mapping = {
            '"': 1,
            "true": 2,
            "false": 3,
        }
        return [[mapping[text]]]

    def decode(self, ids: list[int]) -> str:
        pieces = {
            1: '"',
            2: "true",
            3: "false",
            4: "7",
            5: "word",
            6: "\n",
        }
        return pieces[ids[0]]


def test_token_constraints_classify_string_and_number_tokens(
    tmp_path: Path,
) -> None:
    """Token constraints are built from the model vocabulary."""
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(
        json.dumps({'"': 1, "true": 2, "false": 3, "7": 4, "word": 5, "\n": 6}),
        encoding="utf-8",
    )

    constraints = TokenConstraints.from_model(FakeVocabModel(vocab_path))

    assert constraints.quote_ids == [1]
    assert constraints.true_ids == [2]
    assert constraints.false_ids == [3]
    assert 4 in constraints.number_ids
    assert 5 in constraints.safe_string_ids
    assert 6 not in constraints.safe_string_ids
