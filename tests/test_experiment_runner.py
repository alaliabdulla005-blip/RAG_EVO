"""Tests for src/rag_evo/runner/experiment_runner.py — no real API calls."""

import re
from pathlib import Path
from unittest.mock import MagicMock

from rag_evo.cache.cache import ResultCache
from rag_evo.config.schema import MicroPilotConfig
from rag_evo.data.schemas import CorpusDocument, QARecord, SupportingFact
from rag_evo.evaluation.harness import (
    AggregateMetrics,
    EvaluationHarness,
    RecordMetrics,
)
from rag_evo.generation.wrapper import GeneratorWrapper, WrapperOutput
from rag_evo.pipeline.rag_pipeline import PipelineResult
from rag_evo.retrieval.bm25_retriever import RetrievalResult
from rag_evo.retrieval.chunker import Chunk
from rag_evo.runner.experiment_runner import (
    ExperimentRunner,
    RunConfig,
    RunResult,
    generate_run_id,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _record(record_id: str = "r001") -> QARecord:
    return QARecord(
        record_id=record_id,
        question="What is the capital of France?",
        gold_answer="Paris",
        supporting_facts=[SupportingFact(doc_title="France", sent_idx=0)],
        source_doc_ids=["france"],
        split="validation",
    )


def _doc() -> CorpusDocument:
    return CorpusDocument(
        doc_id="france",
        title="France",
        sentences=["France is a country.", "Its capital is Paris."],
        text="France is a country. Its capital is Paris.",
    )


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="france__chunk_0000",
        doc_id="france",
        text="France is a country. Its capital is Paris.",
        token_count=8,
        char_start=0,
        char_end=42,
    )


def _wrapper_output() -> WrapperOutput:
    return WrapperOutput(
        answer="Paris",
        input_tokens=100,
        output_tokens=5,
        latency_ms=50.0,
        raw_prompt="Context:\n[DOC_1]: France is a country.\n\nQuestion: ...",
        raw_answer="Paris",
        model_id="test-model",
        provider="groq",
    )


def _fake_aggregate(n: int = 1) -> AggregateMetrics:
    return AggregateMetrics(
        num_records=n,
        mean_em=1.0,
        mean_f1=1.0,
        mean_citation_precision=0.0,
        mean_citation_recall=0.0,
        mean_citation_f1=0.0,
        mean_retrieval_recall_at_k=0.5,
        mean_input_tokens=100.0,
        mean_output_tokens=5.0,
        mean_latency_ms=50.0,
        total_input_tokens=100 * n,
        total_output_tokens=5 * n,
    )


def _fake_record_metrics(record_id: str = "r001") -> RecordMetrics:
    return RecordMetrics(
        record_id=record_id,
        em=1.0,
        f1=1.0,
        citation_precision=0.0,
        citation_recall=0.0,
        citation_f1=0.0,
        retrieval_recall_at_k=0.5,
        input_tokens=100,
        output_tokens=5,
        latency_ms=50.0,
    )


def _fake_pipeline_result(record_id: str = "r001") -> PipelineResult:
    return PipelineResult(
        record_id=record_id,
        question="What is the capital of France?",
        answer="Paris",
        cited_chunk_ids=[],
        retrieved_chunks=[_chunk()],
        input_tokens=100,
        output_tokens=5,
        latency_ms=50.0,
        retriever_type="bm25",
        prompt_strategy="zero-shot",
        chunk_size=256,
        overlap=0.0,
        top_k=5,
        raw_prompt="...",
        raw_answer="Paris",
    )


def _fake_retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        query="What is the capital of France?",
        retrieved_chunks=[(_chunk(), 1.0)],
        top_k=5,
        retriever_type="bm25",
    )


def _make_run_config(
    tmp_path: Path,
    records: list,
    run_id: str = "test_run_001",
    config: MicroPilotConfig | None = None,
) -> RunConfig:
    return RunConfig(
        pipeline_config=config or MicroPilotConfig(),
        run_id=run_id,
        qa_records=records,
        corpus_documents=[_doc()],
        cache_dir=tmp_path / "cache",
        results_dir=tmp_path / "results",
        model_id="test-model",
        seed=42,
    )


def _make_runner(
    tmp_path: Path,
    records: list | None = None,
    mock_generate_return: WrapperOutput | None = None,
) -> tuple[ExperimentRunner, MagicMock, MagicMock, ResultCache]:
    mock_wrapper = MagicMock(spec=GeneratorWrapper)
    mock_wrapper.generate.return_value = mock_generate_return or _wrapper_output()

    mock_harness = MagicMock(spec=EvaluationHarness)
    n = len(records) if records else 1
    mock_harness.evaluate_batch.return_value = (
        [_fake_record_metrics(r.record_id) for r in (records or [_record()])],
        _fake_aggregate(n),
    )

    cache = ResultCache(tmp_path / "cache")
    runner = ExperimentRunner(
        wrapper=mock_wrapper, harness=mock_harness, cache=cache
    )
    return runner, mock_wrapper, mock_harness, cache


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_run_returns_run_result(tmp_path: Path) -> None:
    from unittest.mock import patch

    records = [_record()]
    runner, mock_wrapper, _, _ = _make_runner(tmp_path, records)
    run_config = _make_run_config(tmp_path, records)

    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = _fake_retrieval_result()

    with patch(
        "rag_evo.retrieval.retriever_factory.chunk_corpus",
        return_value=[_chunk()],
    ), patch(
        "rag_evo.retrieval.retriever_factory.BM25Retriever",
        return_value=mock_bm25,
    ):
        result = runner.run(run_config)

    assert isinstance(result, RunResult)
    assert result.run_id == "test_run_001"


def test_run_uses_cache_on_second_call(tmp_path: Path) -> None:
    from unittest.mock import patch

    records = [_record()]
    runner, mock_wrapper, mock_harness, cache = _make_runner(tmp_path, records)

    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = _fake_retrieval_result()

    def _run(run_id: str) -> RunResult:
        return runner.run(
            _make_run_config(tmp_path, records, run_id=run_id)
        )

    with patch(
        "rag_evo.retrieval.retriever_factory.chunk_corpus",
        return_value=[_chunk()],
    ), patch(
        "rag_evo.retrieval.retriever_factory.BM25Retriever",
        return_value=mock_bm25,
    ):
        result1 = _run("run_001")
        # Refresh harness mock for second call
        mock_harness.evaluate_batch.return_value = (
            [_fake_record_metrics()], _fake_aggregate(1)
        )
        result2 = _run("run_002")

    assert mock_wrapper.generate.call_count == 1
    assert result1.cache_misses == 1
    assert result1.cache_hits == 0
    assert result2.cache_hits == 1
    assert result2.cache_misses == 0


def test_run_saves_artifacts(tmp_path: Path) -> None:
    from unittest.mock import patch

    records = [_record()]
    runner, _, _, _ = _make_runner(tmp_path, records)
    run_config = _make_run_config(tmp_path, records)

    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = _fake_retrieval_result()

    with patch(
        "rag_evo.retrieval.retriever_factory.chunk_corpus",
        return_value=[_chunk()],
    ), patch(
        "rag_evo.retrieval.retriever_factory.BM25Retriever",
        return_value=mock_bm25,
    ):
        runner.run(run_config)

    run_dir = tmp_path / "results" / "test_run_001"
    assert run_dir.exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_metrics.json").exists()


def test_run_result_has_correct_cache_stats(tmp_path: Path) -> None:
    from unittest.mock import patch

    records = [_record("r001"), _record("r002"), _record("r003")]
    runner, _, mock_harness, _ = _make_runner(tmp_path, records)

    mock_harness.evaluate_batch.return_value = (
        [_fake_record_metrics(r.record_id) for r in records],
        _fake_aggregate(3),
    )

    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = _fake_retrieval_result()

    run_config = _make_run_config(tmp_path, records)

    with patch(
        "rag_evo.retrieval.retriever_factory.chunk_corpus",
        return_value=[_chunk()],
    ), patch(
        "rag_evo.retrieval.retriever_factory.BM25Retriever",
        return_value=mock_bm25,
    ):
        result = runner.run(run_config)

    assert result.cache_misses == 3
    assert result.cache_hits == 0


def test_generate_run_id_format() -> None:
    cfg = MicroPilotConfig()
    run_id = generate_run_id(cfg)
    # Pattern: YYYYMMDD_HHMMSS_xxxxxxxx (8 hex chars)
    assert re.match(r"^\d{8}_\d{6}_[0-9a-f]{8}$", run_id), (
        f"run_id {run_id!r} does not match expected format"
    )


def test_index_cache_prevents_rebuild(tmp_path: Path) -> None:
    """Two runs with same (chunk_size, overlap, retriever_type) should
    build the retriever only once."""
    from unittest.mock import patch

    records = [_record()]
    runner, mock_wrapper, mock_harness, _ = _make_runner(
        tmp_path, records
    )

    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = _fake_retrieval_result()

    with patch(
        "rag_evo.retrieval.retriever_factory.chunk_corpus",
        return_value=[_chunk()],
    ) as mock_chunk, patch(
        "rag_evo.retrieval.retriever_factory.BM25Retriever",
        return_value=mock_bm25,
    ):
        runner.run(_make_run_config(tmp_path, records, run_id="run_a"))
        # Refresh harness mock for second call
        mock_harness.evaluate_batch.return_value = (
            [_fake_record_metrics()], _fake_aggregate(1)
        )
        runner.run(_make_run_config(tmp_path, records, run_id="run_b"))

    # chunk_corpus called only once — index was reused
    assert mock_chunk.call_count == 1
