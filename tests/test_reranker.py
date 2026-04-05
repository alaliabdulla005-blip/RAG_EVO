"""Tests for src/rag_evo/retrieval/reranker.py — no real models."""

from unittest.mock import MagicMock, patch

import numpy as np

from rag_evo.retrieval.bm25_retriever import RetrievalResult
from rag_evo.retrieval.chunker import Chunk
from rag_evo.retrieval.reranker import Reranker


def _chunk(doc_id: str = "doc", idx: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}__chunk_{idx:04d}",
        doc_id=doc_id,
        text=f"text for {doc_id}",
        token_count=3,
        char_start=0,
        char_end=15,
    )


def _retrieval_result(
    chunks: list[Chunk], retriever_type: str = "bm25"
) -> RetrievalResult:
    return RetrievalResult(
        query="test query",
        retrieved_chunks=[(c, float(i)) for i, c in enumerate(chunks)],
        top_k=len(chunks),
        retriever_type=retriever_type,
    )


def _make_reranker(scores: list[float]) -> Reranker:
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array(scores, dtype=np.float32)
    with patch(
        "rag_evo.retrieval.reranker.CrossEncoder", return_value=mock_ce
    ):
        r = Reranker()
    r._cross_encoder = mock_ce
    return r


def test_reranker_returns_retrieval_result() -> None:
    chunks = [_chunk("a"), _chunk("b")]
    reranker = _make_reranker([0.9, 0.1])
    result = reranker.rerank("query", _retrieval_result(chunks))
    assert isinstance(result, RetrievalResult)


def test_reranker_preserves_retriever_type() -> None:
    chunks = [_chunk("a"), _chunk("b")]
    reranker = _make_reranker([0.5, 0.5])
    result = reranker.rerank("query", _retrieval_result(chunks, "bm25"))
    assert result.retriever_type == "bm25"


def test_reranker_reorders_chunks() -> None:
    chunks = [_chunk("first"), _chunk("second")]
    # Give "second" (index 1) a higher score → it should become first
    reranker = _make_reranker([0.1, 0.9])
    result = reranker.rerank("query", _retrieval_result(chunks))
    assert result.retrieved_chunks[0][0].doc_id == "second"
