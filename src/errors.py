class CallMeMaybeError(Exception):
    """Base exception for expected project errors."""


class FileReadError(CallMeMaybeError):
    """Raised when an input file cannot be read."""


class JsonFormatError(CallMeMaybeError):
    """Raised when a file is not valid JSON."""


class SchemaValidationError(CallMeMaybeError):
    """Raised when JSON data does not match the expected schema."""


class FileWriteError(CallMeMaybeError):
    """Raised when the output file cannot be written."""


class ModelError(CallMeMaybeError):
    """Raised when model interaction fails."""


class SelectionError(CallMeMaybeError):
    """Raised when a function cannot be selected confidently."""


class DecodingError(CallMeMaybeError):
    """Raised when arguments cannot be decoded or validated."""
