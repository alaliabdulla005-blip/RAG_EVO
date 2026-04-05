"""Dense retriever using sentence-transformers embeddings and FAISS index."""

import json
import logging
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_evo.retrieval.bm25_retriever import RetrievalResult
from rag_evo.retrieval.chunker import Chunk

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class DenseRetriever:
    """Dense retriever using sentence-transformers and FAISS.

    Embeds chunks using a sentence-transformers model and retrieves
    the top-k most semantically similar chunks for a given query.

    Args:
        chunks: List of Chunk objects to index. Must be non-empty.
        model_name: Sentence-transformers model name.
        batch_size: Batch size for embedding chunks at index time.

    Raises:
        ValueError: If chunks is empty.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = _DEFAULT_MODEL,
        batch_size: int = 64,
    ) -> None:
        if not chunks:
            raise ValueError("chunks must be non-empty.")
        self._chunks = chunks
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

        logger.info(
            "Embedding %d chunks with model=%s …", len(chunks), model_name
        )
        texts = [c.text for c in chunks]
        embeddings = self._model.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            convert_to_numpy=True,
        )
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        logger.info("FAISS index built with %d vectors (dim=%d).", len(chunks), dim)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve top-k semantically similar chunks for a query.

        Args:
            query: Query string.
            top_k: Number of chunks to return. Must be in [1, 20].

        Returns:
            RetrievalResult with retriever_type="dense".

        Raises:
            ValueError: If top_k not in [1, 20].
            ValueError: If query is empty or whitespace only.
        """
        if not 1 <= top_k <= 20:
            raise ValueError(f"top_k must be in [1, 20], got {top_k}")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace only.")

        q_emb = self._model.encode(
            [query], show_progress_bar=False, convert_to_numpy=True
        ).astype(np.float32)
        faiss.normalize_L2(q_emb)

        k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(q_emb, k)

        retrieved = [
            (self._chunks[int(idx)], float(scores[0][rank]))
            for rank, idx in enumerate(indices[0])
        ]
        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved,
            top_k=top_k,
            retriever_type="dense",
        )


def build_and_save_dense_index(
    chunks: list[Chunk],
    index_dir: Path,
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = 64,
) -> "DenseRetriever":
    """Build a DenseRetriever and save chunks, FAISS index, and metadata.

    Args:
        chunks: List of Chunk objects to index.
        index_dir: Directory to save index files.
        model_name: Sentence-transformers model name.
        batch_size: Embedding batch size.

    Returns:
        The built DenseRetriever ready for use.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    retriever = DenseRetriever(chunks, model_name=model_name, batch_size=batch_size)

    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk)) + "\n")

    faiss.write_index(retriever._index, str(index_dir / "faiss.index"))

    (index_dir / "metadata.json").write_text(
        json.dumps({"model_name": model_name}), encoding="utf-8"
    )
    return retriever


def load_dense_index(
    index_dir: Path,
    batch_size: int = 64,
) -> "DenseRetriever":
    """Load a DenseRetriever from a previously saved index directory.

    Args:
        index_dir: Directory containing chunks.jsonl, faiss.index,
            and metadata.json.
        batch_size: Embedding batch size for rebuild.

    Returns:
        DenseRetriever ready for use.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    for name in ("chunks.jsonl", "faiss.index", "metadata.json"):
        if not (index_dir / name).exists():
            raise FileNotFoundError(f"{name} not found in {index_dir}")

    meta = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    model_name = meta["model_name"]

    chunks: list[Chunk] = []
    with (index_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(Chunk(**json.loads(line)))

    return DenseRetriever(chunks, model_name=model_name, batch_size=batch_size)
