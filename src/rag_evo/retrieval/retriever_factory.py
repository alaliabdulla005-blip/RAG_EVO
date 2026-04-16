"""Factory for building retrievers from configuration parameters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rag_evo.data.schemas import CorpusDocument
from rag_evo.retrieval.bm25_retriever import BM25Retriever, RetrievalResult
from rag_evo.retrieval.chunker import chunk_corpus
from rag_evo.retrieval.dense_retriever import load_dense_index
from rag_evo.retrieval.hybrid_retriever import HybridRetriever
from rag_evo.retrieval.reranker import Reranker

_VALID_TYPES = {"BM25", "dense", "hybrid"}
_DEFAULT_INDEX_ROOT = Path("data/indexes")


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
    index_dir: Path | None = None,
    track: str = "general_qa",
    index_root: Path = _DEFAULT_INDEX_ROOT,
) -> Callable[[str, int], RetrievalResult]:
    """Build and return a retriever callable from config and documents.

    Args:
        config: RetrieverConfig specifying retriever type and params.
        documents: List of CorpusDocument objects to chunk and index.
        reranker: Optional Reranker instance.
        index_dir: Optional explicit path to a pre-built dense FAISS
            index directory. When provided and retriever_type is
            "dense", the index is loaded from disk instead of being
            resolved from ``track``. Ignored for non-dense types.
        track: Track name used to resolve the pre-built dense index
            path when ``index_dir`` is not provided. Valid values:
            ``"general_qa"`` (default) and ``"legal"``. Path format:
            ``{index_root}/dense_{chunk_size}_{overlap}_{track}``.
        index_root: Root directory containing pre-built dense
            indexes. Defaults to ``Path("data/indexes")``.

    Returns:
        Callable with signature retrieve(query, top_k) -> RetrievalResult.

    Raises:
        ValueError: If reranker_enabled is True but reranker is None.
        ValueError: If retriever_type is unknown.
        RuntimeError: If retriever_type is "dense" and the resolved
            pre-built index directory does not exist.
    """
    if config.reranker_enabled and reranker is None:
        raise ValueError(
            "reranker_enabled=True requires a Reranker instance."
        )

    if config.retriever_type == "dense":
        if index_dir is None:
            index_dir = (
                index_root
                / f"dense_{config.chunk_size}_{config.overlap}_{track}"
            )
        if not index_dir.exists():
            raise RuntimeError(
                f"Dense index not found: {index_dir}\n"
                "Run: python scripts/build_dense_indexes.py"
            )
        base = load_dense_index(index_dir)
    else:
        chunks = chunk_corpus(
            documents, chunk_size=config.chunk_size, overlap=config.overlap
        )

        if config.retriever_type == "BM25":
            base = BM25Retriever(chunks)
        elif config.retriever_type == "hybrid":
            base = HybridRetriever(
                chunks,
                alpha=config.hybrid_alpha,
                dense_model_name=config.dense_model_name,
            )
        else:
            raise ValueError(
                f"Unknown retriever_type: {config.retriever_type!r}"
            )

    if config.reranker_enabled and reranker is not None:
        def retrieve(query: str, top_k: int = config.top_k) -> RetrievalResult:
            result = base.retrieve(query, top_k=top_k)
            return reranker.rerank(query, result)
    else:
        def retrieve(query: str, top_k: int = config.top_k) -> RetrievalResult:
            return base.retrieve(query, top_k=top_k)

    return retrieve
