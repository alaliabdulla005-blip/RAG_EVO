"""Tests for src/rag_evo/retrieval/hybrid_retriever.py — no real models."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_evo.retrieval.bm25_retriever import RetrievalResult
from rag_evo.retrieval.chunker import Chunk
from rag_evo.retrieval.hybrid_retriever import HybridRetriever


def _chunk(doc_id: str = "doc", idx: int = 0, text: str = "") -> Chunk:
    t = text or f"text about {doc_id} {idx}"
    return Chunk(
        chunk_id=f"{doc_id}__chunk_{idx:04d}",
        doc_id=doc_id,
        text=t,
        token_count=len(t.split()),
        char_start=0,
        char_end=len(t),
    )


def _five_chunks() -> list[Chunk]:
    return [_chunk(f"doc{i}", i) for i in range(5)]


def _mock_model(dim: int = 8) -> MagicMock:
    mock = MagicMock()
    rng = np.random.default_rng(0)

    def _encode(texts, **kwargs):
        n = len(texts) if isinstance(texts, list) else 1
        return rng.random((n, dim)).astype(np.float32)

    mock.encode.side_effect = _encode
    return mock


@pytest.fixture()
def retriever() -> HybridRetriever:
    chunks = _five_chunks()
    with patch(
        "rag_evo.retrieval.hybrid_retriever.SentenceTransformer",
        return_value=_mock_model(),
    ):
        with patch("rag_evo.retrieval.hybrid_retriever.faiss"):
            r = HybridRetriever(chunks, alpha=0.5)
            # Set embeddings directly since faiss.normalize_L2 was mocked
            r._embeddings = np.random.default_rng(1).random(
                (len(chunks), 8)
            ).astype(np.float32)
    return r


def test_hybrid_returns_retrieval_result(retriever: HybridRetriever) -> None:
    with patch("rag_evo.retrieval.hybrid_retriever.faiss"):
        result = retriever.retrieve("test query", top_k=3)
    assert isinstance(result, RetrievalResult)
    assert result.retriever_type == "hybrid"


def test_hybrid_alpha_validation() -> None:
    chunks = _five_chunks()
    with patch("rag_evo.retrieval.hybrid_retriever.SentenceTransformer"):
        with patch("rag_evo.retrieval.hybrid_retriever.faiss"):
            with pytest.raises(ValueError):
                HybridRetriever(chunks, alpha=1.5)
            with pytest.raises(ValueError):
                HybridRetriever(chunks, alpha=-0.1)


def test_hybrid_top_k_respected(retriever: HybridRetriever) -> None:
    with patch("rag_evo.retrieval.hybrid_retriever.faiss"):
        result = retriever.retrieve("test query", top_k=3)
    assert len(result.retrieved_chunks) == 3


def test_hybrid_pure_bm25_alpha_one() -> None:
    # alpha=1.0 → only BM25 scores contribute
    # Build chunks where one has clear BM25 keyword match
    chunks = [
        _chunk("alpha", 0, "unique rare keyword zxqwerty"),
        _chunk("beta", 1, "common words the and of"),
        _chunk("gamma", 2, "another generic sentence here"),
    ]
    with patch(
        "rag_evo.retrieval.hybrid_retriever.SentenceTransformer",
        return_value=_mock_model(),
    ):
        with patch("rag_evo.retrieval.hybrid_retriever.faiss"):
            r = HybridRetriever(chunks, alpha=1.0)
            r._embeddings = np.zeros((3, 8), dtype=np.float32)

    with patch("rag_evo.retrieval.hybrid_retriever.faiss"):
        result = r.retrieve("unique rare keyword zxqwerty", top_k=3)
    # With alpha=1.0, BM25 dominates — top result should be chunk with keyword
    assert result.retrieved_chunks[0][0].doc_id == "alpha"
