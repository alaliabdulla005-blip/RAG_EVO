"""Tests for src/rag_evo/retrieval/retriever_factory.py — no real models."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from rag_evo.data.schemas import CorpusDocument
from rag_evo.retrieval.bm25_retriever import RetrievalResult
from rag_evo.retrieval.retriever_factory import RetrieverConfig, build_retriever


def _docs() -> list[CorpusDocument]:
    return [
        CorpusDocument(
            doc_id=f"doc{i}",
            title=f"Title {i}",
            sentences=[
                f"This is document number {i}.",
                f"It has content about topic {i}.",
            ],
            text=f"This is document number {i}. It has content about topic {i}.",
        )
        for i in range(3)
    ]


def _mock_model(dim: int = 8) -> MagicMock:
    mock = MagicMock()
    rng = np.random.default_rng(0)

    def _encode(texts, **kwargs):
        n = len(texts) if isinstance(texts, list) else 1
        return rng.random((n, dim)).astype(np.float32)

    mock.encode.side_effect = _encode
    return mock


def test_build_retriever_bm25_returns_callable() -> None:
    config = RetrieverConfig(
        retriever_type="BM25",
        chunk_size=64,
        overlap=0.0,
        top_k=3,
        reranker_enabled=False,
    )
    retrieve = build_retriever(config, _docs())
    result = retrieve("document topic")
    assert isinstance(result, RetrievalResult)
    assert result.retriever_type == "bm25"


def test_build_retriever_reranker_required_raises() -> None:
    config = RetrieverConfig(
        retriever_type="BM25",
        chunk_size=64,
        overlap=0.0,
        top_k=3,
        reranker_enabled=True,
    )
    with pytest.raises(ValueError, match="reranker_enabled=True"):
        build_retriever(config, _docs(), reranker=None)


def test_build_retriever_with_reranker_applies_rerank() -> None:
    config = RetrieverConfig(
        retriever_type="BM25",
        chunk_size=64,
        overlap=0.0,
        top_k=3,
        reranker_enabled=True,
    )
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = MagicMock(spec=RetrievalResult)
    retrieve = build_retriever(config, _docs(), reranker=mock_reranker)
    retrieve("document topic")
    assert mock_reranker.rerank.called


def test_retriever_config_invalid_type_raises() -> None:
    with pytest.raises(ValueError):
        RetrieverConfig(
            retriever_type="unknown",
            chunk_size=64,
            overlap=0.0,
            top_k=3,
            reranker_enabled=False,
        )
