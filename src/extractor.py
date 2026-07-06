import re

NumberValue = int | float

_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_QUOTED_STRING_RE = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def extract_numbers(text: str) -> list[NumberValue]:
    """Extract integer and float literals from text."""
    return [
        float(val) if "." in val else int(val)
        for val in _NUMBER_RE.findall(text)
    ]


def extract_quoted_strings(text: str) -> list[str]:
    """Extract single-quoted and double-quoted strings from text."""
    values: list[str] = []
    for match in _QUOTED_STRING_RE.finditer(text):
        values.append(match.group(1) or match.group(2))
    return values


def extract_booleans(text: str) -> list[bool]:
    """Extract obvious boolean values from text."""
    lowered_words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    values: list[bool] = []
    for word in lowered_words:
        if word in {"true", "yes", "enable", "enabled", "on"}:
            values.append(True)
        elif word in {"false", "no", "disable", "disabled", "off"}:
            values.append(False)
    return values


def extract_last_word(text: str) -> str | None:
    """Extract the last simple word from text."""
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9_-]*\b", text)
    return words[-1] if words else None


def extract_file_path(text: str) -> str | None:
    """Extract Unix or Windows file path from text."""
    unix_match = re.search(r"(/[^\s]+)", text)
    if unix_match:
        return unix_match.group(1)
    windows_match = re.search(r"([A-Za-z]:\\[^\s]+)", text)
    return windows_match.group(1) if windows_match else None


def generate_regex_candidates(text: str) -> list[str]:
    """Generate potential regex candidates based on semantic hints in text."""
    candidates = []
    lowered = text.lower()
    if "number" in lowered or "digit" in lowered:
        candidates.extend([r"\d+", r"[0-9]+"])
    if "vowel" in lowered:
        candidates.extend([r"[aeiouAEIOU]", r"[aeiou]"])
    if "word" in lowered and "'" in text:
        # If the prompt mentions "word" and contains a quoted string,
        # use that quoted string as a regex candidate.
        words = extract_quoted_strings(text)
        if words:
            candidates.append(rf"\b{re.escape(words[0])}\b")
    return candidates


def generate_replacement_candidates(text: str) -> list[str]:
    """Generate symbolic replacement candidates from natural language."""
    lowered = text.lower()
    candidates: list[str] = []

    if "asterisk" in lowered or "asterisks" in lowered:
        candidates.append("*")
    if "space" in lowered:
        candidates.append(" ")
    if "empty" in lowered or "nothing" in lowered:
        candidates.append("")

    return candidates


def extract_word_before_keyword(text: str, keyword: str) -> list[str]:
    """Extract words placed directly before a given keyword."""
    pattern = rf"\b([A-Za-z0-9_-]+)\s+{re.escape(keyword)}\b"
    return re.findall(pattern, text, flags=re.IGNORECASE)


def extract_after_colon(text: str) -> str | None:
    """Extract text after the first colon."""
    if ":" not in text:
        return None
    value = text.split(":", 1)[1].strip()
    return value if value else None
