"""Tests for src/rag_evo/retrieval/chunker.py"""

import pytest

from rag_evo.data.schemas import CorpusDocument
from rag_evo.retrieval.chunker import chunk_corpus, chunk_document


def _doc(text: str, doc_id: str = "test_doc") -> CorpusDocument:
    return CorpusDocument(
        doc_id=doc_id,
        title="Test",
        sentences=[text],
        text=text,
    )


def test_chunk_document_basic() -> None:
    doc = _doc("one two three four five six")
    chunks = chunk_document(doc, chunk_size=64, overlap=0.0)
    # 6 tokens, chunk_size=64 → single chunk (all tokens fit)
    # Use smaller chunk_size via the minimum boundary test instead.
    # Re-test with the actual minimum: chunk_size=64 covers 6 tokens in 1 chunk
    assert len(chunks) == 1
    assert chunks[0].text == "one two three four five six"


def test_chunk_document_small_step() -> None:
    # Use chunk_size=64 (min), but a doc large enough to produce multiple chunks.
    # 130 tokens → 2 chunks with step=64
    tokens = [f"w{i}" for i in range(130)]
    doc = _doc(" ".join(tokens))
    chunks = chunk_document(doc, chunk_size=64, overlap=0.0)
    assert len(chunks) == 3  # 0-63, 64-127, 128-129
    assert chunks[0].text == " ".join(tokens[:64])
    assert chunks[1].text == " ".join(tokens[64:128])


def test_chunk_document_with_overlap() -> None:
    # chunk_size=64, overlap=0.25 → step = int(64 * 0.75) = 48
    tokens = [f"w{i}" for i in range(100)]
    doc = _doc(" ".join(tokens))
    chunks = chunk_document(doc, chunk_size=64, overlap=0.25)
    step = int(64 * 0.75)  # 48
    assert step == 48
    # Second chunk starts at token 48, overlapping tokens 48-63 with first chunk
    assert chunks[1].text == " ".join(tokens[48 : 48 + 64])


def test_chunk_document_empty_text() -> None:
    doc = _doc("")
    assert chunk_document(doc, chunk_size=64) == []


def test_chunk_document_invalid_chunk_size() -> None:
    doc = _doc("hello world " * 10)
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size=10)
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size=2000)


def test_chunk_document_invalid_overlap() -> None:
    doc = _doc("hello world " * 10)
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size=64, overlap=0.8)
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size=64, overlap=-0.1)


def test_chunk_id_format() -> None:
    tokens = [f"w{i}" for i in range(64)]
    doc = _doc(" ".join(tokens), doc_id="test_doc")
    chunks = chunk_document(doc, chunk_size=64, overlap=0.0)
    assert chunks[0].chunk_id == "test_doc__chunk_0000"


def test_chunk_corpus_flat_list() -> None:
    tokens = [f"w{i}" for i in range(64)]
    text = " ".join(tokens)
    docs = [_doc(text, "doc_a"), _doc(text, "doc_b")]
    chunks = chunk_corpus(docs, chunk_size=64, overlap=0.0)
    doc_ids = [c.doc_id for c in chunks]
    assert "doc_a" in doc_ids
    assert "doc_b" in doc_ids
