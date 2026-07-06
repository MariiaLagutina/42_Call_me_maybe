
from typing import Any


def encode_to_ids(model: Any, text: str) -> list[int]:
    """
    Encode text with llm_sdk and normalize the result to list[int].
    The provided SDK returns a 2-D tensor-like object from encode().
    The rest of the project should not depend on torch directly, so this
    helper converts the output into a plain Python list of token ids.
    """
    encoded = model.encode(text)

    if hasattr(encoded, "tolist"):
        raw = encoded.tolist()
    else:
        raw = encoded

    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]

    if not isinstance(raw, list):
        raise TypeError("model.encode() must return a list-like object")

    return [int(token_id) for token_id in raw]


def decode_ids(model: Any, token_ids: list[int]) -> str:
    """
    Decode token ids with llm_sdk.
    The provided SDK returns a string from decode(), but the rest of the
    project should not depend on torch directly, so this helper ensures
    that the output is a plain Python string.
    """
    return str(model.decode(token_ids))
