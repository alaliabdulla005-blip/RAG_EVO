"""Prompt builder for RAG-Evo pipeline.

Constructs system prompts and context blocks from retrieved chunks.
The citation convention is frozen: chunks are labelled [DOC_1], [DOC_2], etc.
"""

from rag_evo.retrieval.chunker import Chunk

SYSTEM_PROMPT_ZERO_SHOT = (
    "You are a helpful assistant. Answer the question based only on the "
    "provided context. You MUST cite the document(s) you used as [DOC_N] "
    "where N is the document number. If the answer is not in the context, "
    "say 'I cannot answer based on the provided context.'"
)

SYSTEM_PROMPT_CHAIN_OF_THOUGHT = (
    "You are a helpful assistant. Answer the question based only on the "
    "provided context. First, identify the relevant information from each "
    "document. Then, reason step by step to reach your answer. "
    "You MUST cite the document(s) you used as [DOC_N] where N is the "
    "document number. If the answer is not in the context, "
    "say 'I cannot answer based on the provided context.'"
)

PROMPT_STRATEGIES = {
    "zero-shot": SYSTEM_PROMPT_ZERO_SHOT,
    "chain-of-thought": SYSTEM_PROMPT_CHAIN_OF_THOUGHT,
}


def build_context_block(chunks: list[Chunk]) -> str:
    """Build the context block from a list of retrieved chunks.

    Formats chunks as:
        [DOC_1]: {chunk_1_text}
        [DOC_2]: {chunk_2_text}
        ...

    Numbering is 1-based and positional. This convention is frozen.

    Args:
        chunks: List of Chunk objects in retrieval rank order. Non-empty.

    Returns:
        Formatted context block string.

    Raises:
        ValueError: If chunks is empty.
    """
    if not chunks:
        raise ValueError("chunks must be non-empty.")
    return "\n".join(
        f"[DOC_{i}]: {chunk.text}" for i, chunk in enumerate(chunks, start=1)
    )


def get_system_prompt(strategy: str = "zero-shot") -> str:
    """Return the system prompt for the given strategy.

    Args:
        strategy: One of "zero-shot" or "chain-of-thought".

    Returns:
        System prompt string.

    Raises:
        ValueError: If strategy is not a known prompt strategy.
    """
    if strategy not in PROMPT_STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}. "
            f"Must be one of: {list(PROMPT_STRATEGIES)}"
        )
    return PROMPT_STRATEGIES[strategy]


def build_prompt_components(
    chunks: list[Chunk],
    query: str,
    strategy: str = "zero-shot",
) -> tuple[str, str]:
    """Build all prompt components for a single RAG call.

    Args:
        chunks: Retrieved chunks in rank order.
        query: The question string.
        strategy: Prompt strategy. One of "zero-shot", "chain-of-thought".

    Returns:
        Tuple of (system_prompt, context_block).

    Raises:
        ValueError: If chunks is empty.
        ValueError: If strategy is unknown.
        ValueError: If query is empty or whitespace only.
    """
    if not chunks:
        raise ValueError("chunks must be non-empty.")
    if not query.strip():
        raise ValueError("query must not be empty or whitespace only.")
    system_prompt = get_system_prompt(strategy)
    context_block = build_context_block(chunks)
    return system_prompt, context_block
