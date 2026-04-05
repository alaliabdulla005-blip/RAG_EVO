"""Document chunker for RAG-Evo retrieval pipeline.

Splits CorpusDocument objects into overlapping chunks suitable for indexing.
Chunk size and overlap are search genes in the GA — they must be parameters,
never hardcoded.
"""

from dataclasses import dataclass

from rag_evo.data.schemas import CorpusDocument


@dataclass(frozen=True)
class Chunk:
    """A single text chunk produced by the chunker.

    Args:
        chunk_id: Unique identifier for this chunk.
            Format: {doc_id}__chunk_{index:04d}
        doc_id: ID of the source CorpusDocument this chunk came from.
        text: The chunk text.
        token_count: Approximate token count using whitespace split.
        char_start: Character offset of this chunk's start in the
            original document text.
        char_end: Character offset of this chunk's end in the
            original document text.
    """

    chunk_id: str
    doc_id: str
    text: str
    token_count: int
    char_start: int
    char_end: int


def chunk_document(
    doc: CorpusDocument,
    chunk_size: int = 256,
    overlap: float = 0.0,
) -> list[Chunk]:
    """Split a CorpusDocument into overlapping token-based chunks.

    Args:
        doc: The CorpusDocument to chunk.
        chunk_size: Number of tokens per chunk. Must be in [64, 1024].
        overlap: Fractional overlap between consecutive chunks.
            Must be in [0.0, 0.5].

    Returns:
        List of Chunk objects. Empty list if doc.text is empty.

    Raises:
        ValueError: If chunk_size not in [64, 1024].
        ValueError: If overlap not in [0.0, 0.5].
    """
    if not 64 <= chunk_size <= 1024:
        raise ValueError(
            f"chunk_size must be in [64, 1024], got {chunk_size}"
        )
    if not 0.0 <= overlap <= 0.5:
        raise ValueError(f"overlap must be in [0.0, 0.5], got {overlap}")

    tokens = doc.text.split()
    if not tokens:
        return []

    # Pre-compute character position of each token
    token_positions: list[int] = []
    char_pos = 0
    for token in tokens:
        token_positions.append(char_pos)
        char_pos += len(token) + 1  # +1 for space

    step = max(1, int(chunk_size * (1 - overlap)))
    chunks: list[Chunk] = []

    for index, start in enumerate(range(0, len(tokens), step)):
        window = tokens[start : start + chunk_size]
        text = " ".join(window)
        char_start = token_positions[start]
        last_idx = min(start + chunk_size, len(tokens)) - 1
        char_end = token_positions[last_idx] + len(tokens[last_idx])
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}__chunk_{index:04d}",
                doc_id=doc.doc_id,
                text=text,
                token_count=len(window),
                char_start=char_start,
                char_end=char_end,
            )
        )

    return chunks


def chunk_corpus(
    documents: list[CorpusDocument],
    chunk_size: int = 256,
    overlap: float = 0.0,
) -> list[Chunk]:
    """Chunk all documents in a corpus.

    Args:
        documents: List of CorpusDocument objects to chunk.
        chunk_size: Passed to chunk_document.
        overlap: Passed to chunk_document.

    Returns:
        Flat list of all chunks from all documents, in document order.
    """
    result: list[Chunk] = []
    for doc in documents:
        result.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    return result
