"""BM25 retriever for RAG-Evo.

Builds a BM25 index over a list of Chunk objects and retrieves
the top-k most relevant chunks for a given query.

Uses bm25s (sparse-matrix BM25) for fast, GIL-friendly scoring.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import bm25s

from rag_evo.retrieval.chunker import Chunk


@dataclass(frozen=True)
class RetrievalResult:
    """A single retrieval result for one query.

    Args:
        query: The original query string.
        retrieved_chunks: List of (chunk, score) tuples ranked by
            descending BM25 score. Length == top_k.
        top_k: The number of chunks requested.
        retriever_type: Always "bm25" for this module.
    """

    query: str
    retrieved_chunks: list[tuple[Chunk, float]]
    top_k: int
    retriever_type: str


class BM25Retriever:
    """BM25 retriever built over a list of Chunk objects.

    Args:
        chunks: List of Chunk objects to index. Must be non-empty.

    Raises:
        ValueError: If chunks is empty.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("chunks must be non-empty.")
        self._chunks = chunks
        corpus_tokens = bm25s.tokenize(
            [c.text for c in chunks], show_progress=False
        )
        self._bm25 = bm25s.BM25(k1=1.5, b=0.75)
        self._bm25.index(corpus_tokens, show_progress=False)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve the top-k most relevant chunks for a query.

        Args:
            query: The query string. Tokenised by whitespace internally.
            top_k: Number of chunks to return. Must be in [1, 20].

        Returns:
            RetrievalResult with retrieved_chunks ranked by descending score.

        Raises:
            ValueError: If top_k not in [1, 20].
            ValueError: If query is empty or whitespace only.
        """
        if not 1 <= top_k <= 20:
            raise ValueError(f"top_k must be in [1, 20], got {top_k}")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace only.")

        query_tokens = bm25s.tokenize([query], show_progress=False)
        k = min(top_k, len(self._chunks))
        results, scores = self._bm25.retrieve(
            query_tokens, k=k, show_progress=False
        )
        indices = results[0]
        score_vals = scores[0]
        retrieved: list[tuple[Chunk, float]] = [
            (self._chunks[int(indices[i])], float(score_vals[i]))
            for i in range(k)
        ]
        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved,
            top_k=top_k,
            retriever_type="bm25",
        )


def build_and_save_index(
    chunks: list[Chunk],
    index_dir: Path,
) -> BM25Retriever:
    """Build a BM25Retriever and save the chunk list to disk.

    Saves chunks as chunks.jsonl in index_dir. Does NOT pickle the BM25
    object — rebuilds from chunks on load.

    Args:
        chunks: List of Chunk objects to index.
        index_dir: Directory to save index files.

    Returns:
        The built BM25Retriever ready for use.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk)) + "\n")
    return BM25Retriever(chunks)


def load_index(index_dir: Path) -> BM25Retriever:
    """Load a BM25Retriever from a previously saved index directory.

    Loads chunks.jsonl and rebuilds the BM25 index from scratch.

    Args:
        index_dir: Directory containing chunks.jsonl.

    Returns:
        BM25Retriever ready for use.

    Raises:
        FileNotFoundError: If chunks.jsonl does not exist.
    """
    chunks_path = index_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found in {index_dir}")
    chunks: list[Chunk] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(Chunk(**json.loads(line)))
    return BM25Retriever(chunks)
