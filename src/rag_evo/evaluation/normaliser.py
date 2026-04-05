"""Answer normalisation for evaluation metrics.

Normalisation is applied before EM and F1 computation.
The normalisation pipeline is frozen — never change it.
"""

import re
import string

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalise_answer(text: str) -> str:
    """Normalise an answer string for metric computation.

    Pipeline (applied in order):
    1. Lowercase
    2. Remove punctuation (all chars in string.punctuation)
    3. Collapse whitespace (multiple spaces -> single space)
    4. Strip leading and trailing whitespace

    Args:
        text: Raw answer string from model or gold label.

    Returns:
        Normalised string. May be empty if input was punctuation only.

    Examples:
        >>> normalise_answer("Yes!")
        'yes'
        >>> normalise_answer("  New   York  ")
        'new york'
        >>> normalise_answer("U.S.A.")
        'usa'
    """
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def tokenise(text: str) -> list[str]:
    """Tokenise a normalised answer string by whitespace.

    Args:
        text: Normalised answer string (output of normalise_answer).

    Returns:
        List of tokens. Empty list if text is empty.
    """
    if not text:
        return []
    return text.split()
