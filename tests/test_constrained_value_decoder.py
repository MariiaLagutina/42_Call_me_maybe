import json
from pathlib import Path

from src.constrained_value_decoder import ConstrainedValueDecoder
from src.token_constraints import TokenConstraints


class FakeConstrainedModel:
    """Fake model that makes token-mask decisions observable."""

    def __init__(self, vocab_path: Path) -> None:
        self.vocab_path = vocab_path
        self.string_calls = 0

    def get_path_to_vocab_file(self) -> str:
        return str(self.vocab_path)

    def encode(self, text: str) -> list[list[int]]:
        mapping = {
            '"': 1,
            "true": 2,
            "false": 3,
        }
        if text in mapping:
            return [[mapping[text]]]
        if text.startswith("string"):
            return [[8]]
        return [[0]]

    def decode(self, ids: list[int]) -> str:
        pieces = {
            0: "<ctx>",
            1: '"',
            2: "true",
            3: "false",
            4: "5",
            5: "word",
            6: 'BAD"',
            8: "<string>",
        }
        return pieces[ids[0]]

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        logits = [0.0] * 9
        logits[6] = 100.0

        if 8 in input_ids:
            if input_ids[-1] == 5:
                logits[1] = 20.0
            else:
                logits[5] = 12.0
        else:
            logits[2] = 10.0
            logits[4] = 15.0

        return logits


def make_decoder(tmp_path: Path) -> ConstrainedValueDecoder:
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(
        json.dumps(
            {
                "<ctx>": 0,
                '"': 1,
                "true": 2,
                "false": 3,
                "5": 4,
                "word": 5,
                'BAD"': 6,
                "<string>": 8,
            }
        ),
        encoding="utf-8",
    )
    model = FakeConstrainedModel(vocab_path)
    return ConstrainedValueDecoder(model, TokenConstraints.from_model(model))


def test_choose_boolean_scores_only_boolean_literals(tmp_path: Path) -> None:
    """Boolean fallback is constrained to true/false tokens."""
    decoder = make_decoder(tmp_path)

    assert decoder.choose_boolean("JSON value: ") is True


def test_generate_number_masks_non_numeric_tokens(tmp_path: Path) -> None:
    """Number fallback cannot select non-numeric tokens."""
    decoder = make_decoder(tmp_path)

    assert decoder.generate_number("JSON value: ", integer_only=True) == 5


def test_generate_string_masks_bad_tokens_and_closes(tmp_path: Path) -> None:
    """String fallback cannot select disallowed high-logit tokens."""
    decoder = make_decoder(tmp_path)

    assert decoder.generate_string("string value: ") == "word"
