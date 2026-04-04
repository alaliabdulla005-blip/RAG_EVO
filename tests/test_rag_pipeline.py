"""Tests for src/rag_evo/pipeline/rag_pipeline.py"""

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rag_evo.data.schemas import QARecord, SupportingFact
from rag_evo.generation.wrapper import WrapperOutput
from rag_evo.pipeline.rag_pipeline import (
    PipelineResult,
    RAGPipeline,
    load_results,
    save_results,
)
from rag_evo.retrieval.bm25_retriever import RetrievalResult
from rag_evo.retrieval.chunker import Chunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(text: str = "Some text here", idx: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"doc__chunk_{idx:04d}",
        doc_id="doc",
        text=text,
        token_count=len(text.split()),
        char_start=0,
        char_end=len(text),
    )


def _qa_record(record_id: str = "r1") -> QARecord:
    return QARecord(
        record_id=record_id,
        question="What is the capital of France?",
        gold_answer="Paris",
        supporting_facts=[SupportingFact(doc_title="France", sent_idx=0)],
        source_doc_ids=["france"],
        split="validation",
    )


def _wrapper_output(answer: str = "Paris. [DOC_1]") -> WrapperOutput:
    return WrapperOutput(
        answer=answer,
        input_tokens=80,
        output_tokens=10,
        latency_ms=200.0,
        raw_prompt="Context:\nfoo\n\nQuestion: bar",
        raw_answer=answer,
        model_id="test-model",
        provider="groq",
    )


def _mock_retriever(chunks: list[Chunk] | None = None) -> MagicMock:
    if chunks is None:
        chunks = [_chunk(f"text {i}", i) for i in range(3)]
    retrieval = RetrievalResult(
        query="q",
        retrieved_chunks=[(c, float(3 - i)) for i, c in enumerate(chunks)],
        top_k=len(chunks),
        retriever_type="bm25",
    )
    mock = MagicMock()
    mock.retrieve.return_value = retrieval
    return mock


def _mock_wrapper(answer: str = "Paris. [DOC_1]") -> MagicMock:
    mock = MagicMock()
    mock.generate.return_value = _wrapper_output(answer)
    return mock


def _pipeline(
    retriever: MagicMock | None = None,
    wrapper: MagicMock | None = None,
    answer: str = "Paris. [DOC_1]",
) -> RAGPipeline:
    return RAGPipeline(
        retriever=retriever or _mock_retriever(),
        wrapper=wrapper or _mock_wrapper(answer),
        chunk_size=256,
        overlap=0.0,
        top_k=3,
        prompt_strategy="zero-shot",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_run_returns_pipeline_result() -> None:
    result = _pipeline().run(_qa_record())
    assert isinstance(result, PipelineResult)
    assert result.cited_chunk_ids == ["[DOC_1]"]


def test_pipeline_result_is_frozen() -> None:
    result = _pipeline().run(_qa_record())
    with pytest.raises(FrozenInstanceError):
        result.answer = "changed"  # type: ignore[misc]


def test_pipeline_run_batch_returns_list() -> None:
    pipeline = _pipeline()
    records = [_qa_record(str(i)) for i in range(3)]
    results = pipeline.run_batch(records)
    assert len(results) == 3


def test_pipeline_run_batch_continues_on_error() -> None:
    call_count = 0

    def _side_effect(*args: object, **kwargs: object) -> WrapperOutput:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated failure")
        return _wrapper_output()

    wrapper = MagicMock()
    wrapper.generate.side_effect = _side_effect
    pipeline = _pipeline(wrapper=wrapper)
    records = [_qa_record(str(i)) for i in range(3)]
    results = pipeline.run_batch(records)
    assert len(results) == 2


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    pipeline = _pipeline()
    r1 = pipeline.run(_qa_record("r1"))
    r2 = pipeline.run(_qa_record("r2"))
    out = tmp_path / "results.jsonl"
    save_results([r1, r2], out)
    loaded = load_results(out)
    assert loaded[0].question == r1.question
    assert loaded[1].answer == r2.answer


def test_cited_chunk_extraction() -> None:
    pipeline = _pipeline(answer="Based on [DOC_1] and [DOC_2], the answer is X.")
    result = pipeline.run(_qa_record())
    assert result.cited_chunk_ids == ["[DOC_1]", "[DOC_2]"]
