"""Hybrid retriever combining BM25 and dense retrieval scores."""

import logging

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from rag_evo.retrieval.bm25_retriever import RetrievalResult
from rag_evo.retrieval.chunker import Chunk

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _minmax_normalise(scores: np.ndarray) -> np.ndarray:
    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - mn) / (mx - mn)).astype(np.float32)


class HybridRetriever:
    """Hybrid retriever combining BM25 and dense scores via weighted sum.

    hybrid_score = alpha * bm25_norm + (1 - alpha) * dense_norm

    Args:
        chunks: List of Chunk objects. Must be non-empty.
        alpha: Weight for BM25 score. Must be in [0.0, 1.0].
        dense_model_name: Sentence-transformers model for dense retrieval.

    Raises:
        ValueError: If chunks is empty.
        ValueError: If alpha not in [0.0, 1.0].
    """

    def __init__(
        self,
        chunks: list[Chunk],
        alpha: float = 0.5,
        dense_model_name: str = _DEFAULT_MODEL,
    ) -> None:
        if not chunks:
            raise ValueError("chunks must be non-empty.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")

        self._chunks = chunks
        self._alpha = alpha
        tokenised = [c.text.split() for c in chunks]
        self._bm25 = BM25Okapi(tokenised)

        logger.info(
            "Embedding %d chunks for hybrid retriever …", len(chunks)
        )
        self._model = SentenceTransformer(dense_model_name)
        embeddings = self._model.encode(
            [c.text for c in chunks],
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        # Store raw embeddings (not normalised) — we compute dot product
        # after per-query normalisation for cosine similarity
        faiss.normalize_L2(embeddings)
        self._embeddings = embeddings

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve top-k chunks by combined BM25 and dense score.

        Args:
            query: Query string.
            top_k: Number of chunks to return. Must be in [1, 20].

        Returns:
            RetrievalResult with retriever_type="hybrid".
        """
        if not 1 <= top_k <= 20:
            raise ValueError(f"top_k must be in [1, 20], got {top_k}")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace only.")

        bm25_scores = np.array(
            self._bm25.get_scores(query.split()), dtype=np.float32
        )

        q_emb = self._model.encode(
            [query], show_progress_bar=False, convert_to_numpy=True
        ).astype(np.float32)
        faiss.normalize_L2(q_emb)
        dense_scores = (self._embeddings @ q_emb.T).squeeze(-1)

        bm25_norm = _minmax_normalise(bm25_scores)
        dense_norm = _minmax_normalise(dense_scores)

        hybrid = self._alpha * bm25_norm + (1.0 - self._alpha) * dense_norm
        order = np.argsort(hybrid)[::-1]

        k = min(top_k, len(self._chunks))
        retrieved = [
            (self._chunks[int(i)], float(hybrid[int(i)]))
            for i in order[:k]
        ]
        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved,
            top_k=top_k,
            retriever_type="hybrid",
        )
