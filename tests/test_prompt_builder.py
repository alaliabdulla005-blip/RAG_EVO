"""Tests for src/rag_evo/pipeline/prompt_builder.py"""

import pytest

from rag_evo.pipeline.prompt_builder import (
    SYSTEM_PROMPT_CHAIN_OF_THOUGHT,
    SYSTEM_PROMPT_ZERO_SHOT,
    build_context_block,
    build_prompt_components,
    get_system_prompt,
)
from rag_evo.retrieval.chunker import Chunk


def _chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"doc__chunk_{idx:04d}",
        doc_id="doc",
        text=text,
        token_count=len(text.split()),
        char_start=0,
        char_end=len(text),
    )


def test_build_context_block_format() -> None:
    chunks = [_chunk("Alpha text", 0), _chunk("Beta text", 1), _chunk("Gamma text", 2)]
    result = build_context_block(chunks)
    assert "[DOC_1]: Alpha text" in result
    assert "[DOC_2]: Beta text" in result
    assert "[DOC_3]: Gamma text" in result
    assert result.startswith("[DOC_1]")


def test_build_context_block_empty_raises() -> None:
    with pytest.raises(ValueError):
        build_context_block([])


def test_get_system_prompt_zero_shot() -> None:
    assert get_system_prompt("zero-shot") == SYSTEM_PROMPT_ZERO_SHOT


def test_get_system_prompt_chain_of_thought() -> None:
    assert get_system_prompt("chain-of-thought") == SYSTEM_PROMPT_CHAIN_OF_THOUGHT


def test_get_system_prompt_invalid_raises() -> None:
    with pytest.raises(ValueError):
        get_system_prompt("unknown")


def test_build_prompt_components_returns_tuple() -> None:
    chunks = [_chunk("Some context", 0)]
    system_prompt, context_block = build_prompt_components(
        chunks, "What is this?", "zero-shot"
    )
    assert isinstance(system_prompt, str) and system_prompt
    assert "[DOC_1]" in context_block


def test_build_prompt_components_empty_query_raises() -> None:
    chunks = [_chunk("Some context", 0)]
    with pytest.raises(ValueError):
        build_prompt_components(chunks, "   ")
