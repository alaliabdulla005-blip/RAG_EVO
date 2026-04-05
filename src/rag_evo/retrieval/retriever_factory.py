"""Factory for building retrievers from configuration parameters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rag_evo.data.schemas import CorpusDocument
from rag_evo.retrieval.bm25_retriever import BM25Retriever, RetrievalResult
from rag_evo.retrieval.chunker import chunk_corpus
from rag_evo.retrieval.dense_retriever import DenseRetriever
from rag_evo.retrieval.hybrid_retriever import HybridRetriever
from rag_evo.retrieval.reranker import Reranker

_VALID_TYPES = {"BM25", "dense", "hybrid"}


@dataclass(frozen=True)
class RetrieverConfig:
    """Configuration for retriever construction.

    Args:
        retriever_type: One of "BM25", "dense", "hybrid".
        chunk_size: Chunk size used to build the chunk list.
        overlap: Overlap used to build the chunk list.
        top_k: Number of chunks to retrieve per query.
        reranker_enabled: Whether to apply cross-encoder reranking.
        dense_model_name: Model for dense/hybrid retrieval.
        hybrid_alpha: BM25 weight for hybrid retrieval.
    """

    retriever_type: str
    chunk_size: int
    overlap: float
    top_k: int
    reranker_enabled: bool
    dense_model_name: str = "all-MiniLM-L6-v2"
    hybrid_alpha: float = 0.5

    def __post_init__(self) -> None:
        if self.retriever_type not in _VALID_TYPES:
            raise ValueError(
                f"retriever_type must be one of {_VALID_TYPES}, "
                f"got {self.retriever_type!r}"
            )


def build_retriever(
    config: RetrieverConfig,
    documents: list[CorpusDocument],
    reranker: Reranker | None = None,
) -> Callable[[str, int], RetrievalResult]:
    """Build and return a retriever callable from config and documents.

    Args:
        config: RetrieverConfig specifying retriever type and params.
        documents: List of CorpusDocument objects to chunk and index.
        reranker: Optional Reranker instance.

    Returns:
        Callable with signature retrieve(query, top_k) -> RetrievalResult.

    Raises:
        ValueError: If reranker_enabled is True but reranker is None.
        ValueError: If retriever_type is unknown.
    """
    if config.reranker_enabled and reranker is None:
        raise ValueError(
            "reranker_enabled=True requires a Reranker instance."
        )

    chunks = chunk_corpus(
        documents, chunk_size=config.chunk_size, overlap=config.overlap
    )

    if config.retriever_type == "BM25":
        base = BM25Retriever(chunks)
    elif config.retriever_type == "dense":
        base = DenseRetriever(chunks, model_name=config.dense_model_name)
    elif config.retriever_type == "hybrid":
        base = HybridRetriever(
            chunks,
            alpha=config.hybrid_alpha,
            dense_model_name=config.dense_model_name,
        )
    else:
        raise ValueError(f"Unknown retriever_type: {config.retriever_type!r}")

    if config.reranker_enabled and reranker is not None:
        def retrieve(query: str, top_k: int = config.top_k) -> RetrievalResult:
            result = base.retrieve(query, top_k=top_k)
            return reranker.rerank(query, result)
    else:
        def retrieve(query: str, top_k: int = config.top_k) -> RetrievalResult:
            return base.retrieve(query, top_k=top_k)

    return retrieve
