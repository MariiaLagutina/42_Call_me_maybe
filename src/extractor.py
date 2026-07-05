import re


NumberValue = int | float


_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_QUOTED_STRING_RE = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def extract_numbers(text: str) -> list[NumberValue]:
    """Extract integer and float literals from text."""
    values: list[NumberValue] = []

    for match in _NUMBER_RE.finditer(text):
        raw_value = match.group(0)

        if "." in raw_value:
            values.append(float(raw_value))
        else:
            values.append(int(raw_value))

    return values


def extract_quoted_strings(text: str) -> list[str]:
    """Extract single-quoted and double-quoted strings from text."""
    values: list[str] = []

    for match in _QUOTED_STRING_RE.finditer(text):
        single_quoted = match.group(1)
        double_quoted = match.group(2)

        if single_quoted is not None:
            values.append(single_quoted)
        elif double_quoted is not None:
            values.append(double_quoted)

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
    words: list[str] = re.findall(r"\b[A-Za-z][A-Za-z0-9_-]*\b", text)

    if not words:
        return None

    return words[-1]


def extract_replacement_word(text: str) -> str | None:
    """Extract a likely replacement word from substitution-style prompts."""
    patterns = [
        r"\bwith\s+(asterisks|dash|underscore|space)\b",
        r"\bwith\s+([A-Za-z0-9_*.-]+)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            value = match.group(1)
            lowered_value = value.lower()

            if lowered_value == "asterisks":
                return "*"
            if lowered_value == "dash":
                return "-"
            if lowered_value == "underscore":
                return "_"
            if lowered_value == "space":
                return " "

            return value

    return None


def extract_regex_pattern(text: str) -> str | None:
    """Extract an obvious regex pattern from a substitution-style prompt."""
    lowered = text.lower()

    if "all numbers" in lowered or "digits" in lowered:
        return r"\d+"

    if "all vowels" in lowered or "vowels" in lowered:
        return r"[aeiouAEIOU]"

    return None


def extract_substitution_source(text: str) -> str | None:
    """Extract source string from substitution-style prompts."""
    quoted = extract_quoted_strings(text)

    if len(quoted) >= 3 and re.search(r"\bin\b", text, flags=re.IGNORECASE):
        return quoted[-1]

    if quoted:
        return quoted[0]

    return None


def extract_substitution_target(text: str) -> str | None:
    """Extract target regex from substitution-style prompts."""
    quoted = extract_quoted_strings(text)
    lowered = text.lower()

    if "substitute" in lowered and quoted:
        return rf"\b{re.escape(quoted[0])}\b"

    return None


def extract_substitution_replacement(text: str) -> str | None:
    """Extract replacement value from substitution-style prompts."""
    quoted = extract_quoted_strings(text)
    lowered = text.lower()

    if "substitute" in lowered and len(quoted) >= 2:
        return quoted[1]

    return extract_replacement_word(text)


def extract_database_name(text: str) -> str | None:
    """Extract database name from prompts like 'on the production database'."""
    patterns = [
        r"\bon\s+the\s+([A-Za-z0-9_-]+)\s+database\b",
        r"\bon\s+([A-Za-z0-9_-]+)\s+database\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def extract_file_path(text: str) -> str | None:
    """Extract Unix or Windows file path from text."""
    unix_match = re.search(r"(/[^\s]+)", text)
    if unix_match:
        return unix_match.group(1)

    windows_match = re.search(r"([A-Za-z]:\\[^\s]+)", text)
    if windows_match:
        return windows_match.group(1)

    return None


def extract_encoding(text: str) -> str | None:
    """Extract encoding from prompts like 'with utf-8 encoding'."""
    match = re.search(
        r"\bwith\s+([A-Za-z0-9_-]+)\s+encoding\b",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return None


def extract_template(text: str) -> str | None:
    """Extract template after 'Format template:'."""
    match = re.search(
        r"\bFormat\s+template:\s*(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    return None
