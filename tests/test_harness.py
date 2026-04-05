"""Tests for src/rag_evo/evaluation/harness.py"""

from pathlib import Path

import pytest

from rag_evo.data.schemas import QARecord, SupportingFact
from rag_evo.evaluation.harness import (
    AggregateMetrics,
    EvaluationHarness,
    RecordMetrics,
)
from rag_evo.pipeline.rag_pipeline import PipelineResult
from rag_evo.retrieval.chunker import Chunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(doc_id: str = "scott_derrickson", idx: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}__chunk_{idx:04d}",
        doc_id=doc_id,
        text="Scott Derrickson is an American director.",
        token_count=6,
        char_start=0,
        char_end=41,
    )


def _qa_record(
    record_id: str = "r1",
    gold_answer: str = "yes",
    source_doc_ids: list[str] | None = None,
) -> QARecord:
    return QARecord(
        record_id=record_id,
        question="Were they the same nationality?",
        gold_answer=gold_answer,
        supporting_facts=[
            SupportingFact(doc_title="Scott Derrickson", sent_idx=0),
        ],
        source_doc_ids=source_doc_ids or ["scott_derrickson"],
        split="validation",
    )


def _pipeline_result(
    record_id: str = "r1",
    answer: str = "Yes. [DOC_1]",
    doc_id: str = "scott_derrickson",
) -> PipelineResult:
    return PipelineResult(
        record_id=record_id,
        question="Were they the same nationality?",
        answer=answer,
        cited_chunk_ids=["[DOC_1]"],
        retrieved_chunks=[_chunk(doc_id)],
        input_tokens=100,
        output_tokens=10,
        latency_ms=200.0,
        retriever_type="bm25",
        prompt_strategy="zero-shot",
        chunk_size=256,
        overlap=0.0,
        top_k=5,
        raw_prompt="Context:\nfoo\n\nQuestion: bar",
        raw_answer=answer,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_evaluate_result_returns_record_metrics() -> None:
    harness = EvaluationHarness([_qa_record()])
    m = harness.evaluate_result(_pipeline_result())
    assert isinstance(m, RecordMetrics)


def test_evaluate_result_em_correct_answer() -> None:
    harness = EvaluationHarness([_qa_record(gold_answer="yes")])
    m = harness.evaluate_result(_pipeline_result(answer="Yes. [DOC_1]"))
    # normalise_answer("Yes. [DOC_1]") → "yes doc_1"
    # normalise_answer("yes") → "yes"
    # These are NOT equal, so EM should be 0.0.
    # For EM=1.0, the model must output only "yes".
    assert m.em == 0.0


def test_evaluate_result_em_exact() -> None:
    harness = EvaluationHarness([_qa_record(gold_answer="yes")])
    m = harness.evaluate_result(_pipeline_result(answer="yes"))
    assert m.em == 1.0


def test_evaluate_result_em_wrong_answer() -> None:
    harness = EvaluationHarness([_qa_record(gold_answer="no")])
    m = harness.evaluate_result(_pipeline_result(answer="yes"))
    assert m.em == 0.0


def test_evaluate_result_citation_maps_positions() -> None:
    harness = EvaluationHarness(
        [_qa_record(source_doc_ids=["scott_derrickson"])]
    )
    m = harness.evaluate_result(
        _pipeline_result(answer="Yes [DOC_1]", doc_id="scott_derrickson")
    )
    assert m.citation_recall == 1.0


def test_evaluate_batch_returns_aggregate() -> None:
    records = [_qa_record("r1"), _qa_record("r2")]
    harness = EvaluationHarness(records)
    results = [_pipeline_result("r1"), _pipeline_result("r2")]
    per_record, agg = harness.evaluate_batch(results)
    assert isinstance(agg, AggregateMetrics)
    assert agg.num_records == 2
    assert len(per_record) == 2


def test_save_metrics_creates_files(tmp_path: Path) -> None:
    harness = EvaluationHarness([_qa_record()])
    per_record, agg = harness.evaluate_batch([_pipeline_result()])
    harness.save_metrics(per_record, agg, tmp_path)
    assert (tmp_path / "per_record_metrics.jsonl").exists()
    assert (tmp_path / "aggregate_metrics.json").exists()


def test_missing_record_id_raises() -> None:
    harness = EvaluationHarness([_qa_record("r1")])
    with pytest.raises(KeyError):
        harness.evaluate_result(_pipeline_result("unknown"))
