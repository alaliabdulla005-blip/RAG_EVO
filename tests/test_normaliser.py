"""Tests for src/rag_evo/evaluation/normaliser.py"""

from rag_evo.evaluation.normaliser import normalise_answer, tokenise


def test_normalise_lowercase() -> None:
    assert normalise_answer("YES") == "yes"


def test_normalise_strips_punctuation() -> None:
    assert normalise_answer("yes!") == "yes"
    assert normalise_answer("U.S.A.") == "usa"


def test_normalise_collapses_whitespace() -> None:
    assert normalise_answer("new  york") == "new york"


def test_normalise_strips_edges() -> None:
    assert normalise_answer("  yes  ") == "yes"


def test_normalise_empty() -> None:
    assert normalise_answer("") == ""


def test_normalise_punctuation_only() -> None:
    assert normalise_answer("!!!") == ""


def test_tokenise_basic() -> None:
    assert tokenise("new york") == ["new", "york"]


def test_tokenise_empty() -> None:
    assert tokenise("") == []
