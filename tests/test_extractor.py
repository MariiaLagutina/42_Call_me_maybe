from src.extractor import (
    extract_booleans,
    extract_file_path,
    extract_last_word,
    extract_numbers,
    extract_quoted_strings,
    generate_regex_candidates,
)


def test_extract_numbers_supports_ints_floats_and_signs() -> None:
    """Number extraction keeps integer and float values usable."""
    assert extract_numbers("Add -2, 3.5 and +10") == [-2, 3.5, 10]


def test_extract_quoted_strings_supports_single_and_double_quotes() -> None:
    """Quoted string extraction handles both quote styles."""
    assert extract_quoted_strings("Use 'hello' and \"world\"") == [
        "hello",
        "world",
    ]


def test_extract_booleans_from_common_words() -> None:
    """Boolean extraction recognizes common natural-language values."""
    assert extract_booleans("enable cache and disable logging") == [
        True,
        False,
    ]


def test_extract_last_word() -> None:
    """Last-word extraction is used as a fallback for simple names."""
    assert extract_last_word("Greet alice_42") == "alice_42"


def test_extract_file_path_supports_unix_and_windows_paths() -> None:
    """File path extraction returns literal paths without schema knowledge."""
    assert extract_file_path("Read file /tmp/example.txt now") == (
        "/tmp/example.txt"
    )
    assert extract_file_path(r"Read C:\\Users\\me\\file.txt") == (
        r"C:\\Users\\me\\file.txt"
    )


def test_generate_regex_candidates_from_neutral_hints() -> None:
    """Regex candidates come from generic prompt hints."""
    assert generate_regex_candidates("replace every number") == [
        r"\d+",
        r"[0-9]+",
    ]
    assert r"[aeiouAEIOU]" in generate_regex_candidates("match vowels")
