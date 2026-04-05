"""Cross-encoder reranker for RAG-Evo retrieval pipeline.

Applied on top of any retriever as an optional post-processing step.
reranker_enabled=False must be a zero-overhead bypass.
"""

import logging

from sentence_transformers import CrossEncoder

from rag_evo.retrieval.bm25_retriever import RetrievalResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Cross-encoder reranker using sentence-transformers.

    Re-scores retrieved chunks using a cross-encoder model that
    jointly encodes query and passage for more accurate relevance.

    Args:
        model_name: Cross-encoder model name.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._cross_encoder = CrossEncoder(model_name)
        logger.info("Cross-encoder reranker loaded: %s", model_name)

    def rerank(
        self,
        query: str,
        retrieval_result: RetrievalResult,
    ) -> RetrievalResult:
        """Re-score and re-rank chunks in a RetrievalResult.

        Args:
            query: The original query string.
            retrieval_result: Result from any retriever.

        Returns:
            New RetrievalResult with chunks re-ranked by cross-encoder
            score. retriever_type is preserved from the input.
        """
        chunks = [chunk for chunk, _ in retrieval_result.retrieved_chunks]
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._cross_encoder.predict(pairs)

        reranked = sorted(
            zip(chunks, scores.tolist()), key=lambda x: x[1], reverse=True
        )
        return RetrievalResult(
            query=retrieval_result.query,
            retrieved_chunks=list(reranked),
            top_k=retrieval_result.top_k,
            retriever_type=retrieval_result.retriever_type,
        )
