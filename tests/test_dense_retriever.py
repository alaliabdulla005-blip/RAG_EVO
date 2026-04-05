"""Tests for src/rag_evo/retrieval/dense_retriever.py — no real models."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_evo.retrieval.bm25_retriever import RetrievalResult
from rag_evo.retrieval.chunker import Chunk
from rag_evo.retrieval.dense_retriever import DenseRetriever


def _chunk(doc_id: str = "doc", idx: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}__chunk_{idx:04d}",
        doc_id=doc_id,
        text=f"text about {doc_id} topic {idx}",
        token_count=5,
        char_start=0,
        char_end=20,
    )


def _five_chunks() -> list[Chunk]:
    return [_chunk(f"doc{i}", i) for i in range(5)]


def _make_mock_model(n_chunks: int, dim: int = 8) -> MagicMock:
    """Return a mock SentenceTransformer that produces deterministic embeddings."""
    mock = MagicMock()
    rng = np.random.default_rng(42)

    def _encode(texts, **kwargs):
        n = len(texts) if isinstance(texts, list) else 1
        return rng.random((n, dim)).astype(np.float32)

    mock.encode.side_effect = _encode
    return mock


@pytest.fixture()
def retriever() -> DenseRetriever:
    chunks = _five_chunks()
    with patch(
        "rag_evo.retrieval.dense_retriever.SentenceTransformer",
        return_value=_make_mock_model(len(chunks)),
    ):
        return DenseRetriever(chunks)


def test_dense_retriever_returns_retrieval_result(
    retriever: DenseRetriever,
) -> None:
    result = retriever.retrieve("test query", top_k=3)
    assert isinstance(result, RetrievalResult)
    assert result.retriever_type == "dense"


def test_dense_retriever_top_k_respected(retriever: DenseRetriever) -> None:
    result = retriever.retrieve("test query", top_k=3)
    assert len(result.retrieved_chunks) == 3


def test_dense_retriever_empty_chunks_raises() -> None:
    with patch("rag_evo.retrieval.dense_retriever.SentenceTransformer"):
        with pytest.raises(ValueError):
            DenseRetriever([])


def test_dense_retriever_invalid_top_k(retriever: DenseRetriever) -> None:
    with pytest.raises(ValueError):
        retriever.retrieve("query", top_k=0)
    with pytest.raises(ValueError):
        retriever.retrieve("query", top_k=25)


def test_dense_retriever_empty_query_raises(retriever: DenseRetriever) -> None:
    with pytest.raises(ValueError):
        retriever.retrieve("  ", top_k=3)
