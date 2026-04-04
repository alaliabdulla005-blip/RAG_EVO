"""Tests for src/rag_evo/retrieval/bm25_retriever.py"""

from pathlib import Path

import pytest

from rag_evo.retrieval.bm25_retriever import (
    BM25Retriever,
    RetrievalResult,
    build_and_save_index,
    load_index,
)
from rag_evo.retrieval.chunker import Chunk


def _chunk(text: str, idx: int = 0, doc_id: str = "doc") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}__chunk_{idx:04d}",
        doc_id=doc_id,
        text=text,
        token_count=len(text.split()),
        char_start=0,
        char_end=len(text),
    )


def _five_chunks() -> list[Chunk]:
    return [_chunk(f"token{i} content here sample", i) for i in range(5)]


def test_retrieve_returns_retrieval_result() -> None:
    r = BM25Retriever(_five_chunks()).retrieve("token1 content", top_k=3)
    assert isinstance(r, RetrievalResult)
    assert len(r.retrieved_chunks) == 3


def test_retrieve_ranks_by_relevance() -> None:
    chunks = [
        _chunk("Paris is the capital of France", 0, "doc_a"),
        _chunk("The sky is blue and clouds are white", 1, "doc_b"),
        _chunk("Football is a popular sport worldwide", 2, "doc_c"),
    ]
    result = BM25Retriever(chunks).retrieve("capital of France Paris", top_k=3)
    assert result.retrieved_chunks[0][0].doc_id == "doc_a"


def test_retrieve_invalid_top_k() -> None:
    r = BM25Retriever(_five_chunks())
    with pytest.raises(ValueError):
        r.retrieve("query", top_k=0)
    with pytest.raises(ValueError):
        r.retrieve("query", top_k=25)


def test_retrieve_empty_query() -> None:
    with pytest.raises(ValueError):
        BM25Retriever(_five_chunks()).retrieve("   ", top_k=5)


def test_build_and_load_roundtrip(tmp_path: Path) -> None:
    chunks = _five_chunks()
    r1 = build_and_save_index(chunks, tmp_path)
    r2 = load_index(tmp_path)
    res1 = r1.retrieve("token1 content", top_k=1)
    res2 = r2.retrieve("token1 content", top_k=1)
    assert res1.retrieved_chunks[0][0].chunk_id == res2.retrieved_chunks[0][0].chunk_id


def test_empty_chunks_raises() -> None:
    with pytest.raises(ValueError):
        BM25Retriever([])
