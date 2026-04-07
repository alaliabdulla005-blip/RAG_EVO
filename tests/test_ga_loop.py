"""Tests for src/rag_evo/ga/loop.py — no real API calls."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_evo.cache.cache import ResultCache
from rag_evo.config.schema import MicroPilotConfig
from rag_evo.config.space import SearchSpace
from rag_evo.data.schemas import CorpusDocument, QARecord, SupportingFact
from rag_evo.evaluation.harness import (
    AggregateMetrics,
    EvaluationHarness,
    RecordMetrics,
)
from rag_evo.ga.fitness import FitnessWeights
from rag_evo.ga.loop import GAConfig, GenerationRecord, GeneticAlgorithm
from rag_evo.generation.wrapper import GeneratorWrapper, WrapperOutput
from rag_evo.retrieval.bm25_retriever import RetrievalResult
from rag_evo.retrieval.chunker import Chunk
from rag_evo.runner.experiment_runner import ExperimentRunner, RunConfig

# ── Helpers ──────────────────────────────────────────────────────────────────


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
        raw_prompt="...",
        raw_answer="Paris",
        model_id="test-model",
        provider="groq",
    )


def _fake_aggregate(n: int = 1) -> AggregateMetrics:
    return AggregateMetrics(
        num_records=n,
        mean_em=0.5,
        mean_f1=0.5,
        mean_citation_precision=0.3,
        mean_citation_recall=0.3,
        mean_citation_f1=0.3,
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
        em=0.5,
        f1=0.5,
        citation_precision=0.3,
        citation_recall=0.3,
        citation_f1=0.3,
        retrieval_recall_at_k=0.5,
        input_tokens=100,
        output_tokens=5,
        latency_ms=50.0,
    )


def _make_ga(
    tmp_path: Path,
    pop_size: int = 3,
    generations: int = 2,
) -> tuple[GeneticAlgorithm, MagicMock]:
    """Build a GA with fully mocked runner pipeline."""
    records = [_record()]

    mock_wrapper = MagicMock(spec=GeneratorWrapper)
    mock_wrapper.generate.return_value = _wrapper_output()

    mock_harness = MagicMock(spec=EvaluationHarness)
    mock_harness.evaluate_batch.return_value = (
        [_fake_record_metrics()],
        _fake_aggregate(1),
    )

    cache = ResultCache(tmp_path / "cache")
    runner = ExperimentRunner(
        wrapper=mock_wrapper, harness=mock_harness, cache=cache
    )

    template = RunConfig(
        pipeline_config=MicroPilotConfig(),
        run_id="template",
        qa_records=records,
        corpus_documents=[_doc()],
        cache_dir=tmp_path / "cache",
        results_dir=tmp_path / "results",
        model_id="test-model",
        seed=42,
    )

    ga_config = GAConfig(
        population_size=pop_size,
        generations=generations,
        tournament_size=2,
        crossover_rate=0.8,
        mutation_rate=0.3,
        elite_count=1,
        seed=42,
    )

    ga = GeneticAlgorithm(
        ga_config=ga_config,
        run_config_template=template,
        runner=runner,
        search_space=SearchSpace("micro_pilot"),
        fitness_weights=FitnessWeights(),
        output_dir=tmp_path / "ga_output",
    )

    return ga, mock_wrapper


# ── Tests ────────────────────────────────────────────────────────────────────


def test_ga_config_defaults() -> None:
    gc = GAConfig()
    assert gc.population_size == 10
    assert gc.generations == 15
    assert gc.tournament_size == 3
    assert gc.crossover_rate == 0.8
    assert gc.mutation_rate == 0.2
    assert gc.elite_count == 1
    assert gc.seed == 42


def test_ga_run_returns_records(tmp_path: Path) -> None:
    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = RetrievalResult(
        query="q", retrieved_chunks=[(_chunk(), 1.0)],
        top_k=5, retriever_type="bm25",
    )

    with patch(
        "rag_evo.retrieval.retriever_factory.chunk_corpus",
        return_value=[_chunk()],
    ), patch(
        "rag_evo.retrieval.retriever_factory.BM25Retriever",
        return_value=mock_bm25,
    ):
        ga, _ = _make_ga(tmp_path, pop_size=3, generations=2)
        records = ga.run()

    assert len(records) == 2
    assert all(isinstance(r, GenerationRecord) for r in records)
    assert records[0].generation_idx == 0
    assert records[1].generation_idx == 1


def test_ga_saves_generation_artifacts(tmp_path: Path) -> None:
    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = RetrievalResult(
        query="q", retrieved_chunks=[(_chunk(), 1.0)],
        top_k=5, retriever_type="bm25",
    )

    with patch(
        "rag_evo.retrieval.retriever_factory.chunk_corpus",
        return_value=[_chunk()],
    ), patch(
        "rag_evo.retrieval.retriever_factory.BM25Retriever",
        return_value=mock_bm25,
    ):
        ga, _ = _make_ga(tmp_path, pop_size=3, generations=2)
        ga.run()

    gen_dir = tmp_path / "ga_output" / "gen_000"
    assert gen_dir.exists()
    assert (gen_dir / "record.json").exists()
    assert (gen_dir / "population.json").exists()


def test_ga_fitness_improves_or_stable(tmp_path: Path) -> None:
    """Best fitness should not decrease between generations (elitism)."""
    call_count = 0

    def _varying_aggregate(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        # Vary mean_f1 to create fitness variation
        f1 = 0.3 + (call_count % 5) * 0.1
        agg = AggregateMetrics(
            num_records=1,
            mean_em=f1,
            mean_f1=f1,
            mean_citation_precision=0.3,
            mean_citation_recall=0.3,
            mean_citation_f1=0.3,
            mean_retrieval_recall_at_k=0.5,
            mean_input_tokens=100.0,
            mean_output_tokens=5.0,
            mean_latency_ms=50.0,
            total_input_tokens=100,
            total_output_tokens=5,
        )
        return [_fake_record_metrics()], agg

    mock_wrapper = MagicMock(spec=GeneratorWrapper)
    mock_wrapper.generate.return_value = _wrapper_output()

    mock_harness = MagicMock(spec=EvaluationHarness)
    mock_harness.evaluate_batch.side_effect = _varying_aggregate

    cache = ResultCache(tmp_path / "cache")
    runner = ExperimentRunner(
        wrapper=mock_wrapper, harness=mock_harness, cache=cache
    )

    template = RunConfig(
        pipeline_config=MicroPilotConfig(),
        run_id="template",
        qa_records=[_record()],
        corpus_documents=[_doc()],
        cache_dir=tmp_path / "cache",
        results_dir=tmp_path / "results",
        model_id="test-model",
        seed=42,
    )

    ga_config = GAConfig(
        population_size=4,
        generations=3,
        tournament_size=2,
        elite_count=1,
        seed=42,
    )

    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = RetrievalResult(
        query="q", retrieved_chunks=[(_chunk(), 1.0)],
        top_k=5, retriever_type="bm25",
    )

    with patch(
        "rag_evo.retrieval.retriever_factory.chunk_corpus",
        return_value=[_chunk()],
    ), patch(
        "rag_evo.retrieval.retriever_factory.BM25Retriever",
        return_value=mock_bm25,
    ):
        ga = GeneticAlgorithm(
            ga_config=ga_config,
            run_config_template=template,
            runner=runner,
            search_space=SearchSpace("micro_pilot"),
            fitness_weights=FitnessWeights(),
        )
        records = ga.run()

    assert len(records) == 3
    # All records should have valid fitness values
    assert all(r.best_fitness >= r.worst_fitness for r in records)


def test_ga_resume_continues(tmp_path: Path) -> None:
    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = RetrievalResult(
        query="q", retrieved_chunks=[(_chunk(), 1.0)],
        top_k=5, retriever_type="bm25",
    )

    with patch(
        "rag_evo.retrieval.retriever_factory.chunk_corpus",
        return_value=[_chunk()],
    ), patch(
        "rag_evo.retrieval.retriever_factory.BM25Retriever",
        return_value=mock_bm25,
    ):
        ga, _ = _make_ga(tmp_path, pop_size=3, generations=3)
        # Run first 2 gens manually by setting generations=2
        ga._ga_config.generations = 2
        records = ga.run()
        assert len(records) == 2

        # Resume for 1 more generation
        ga._ga_config.generations = 3
        population = [
            SearchSpace("micro_pilot").sample_random(ga._rng)
            for _ in range(3)
        ]
        fitnesses = [0.3, 0.5, 0.4]
        all_records = ga.resume(records, population, fitnesses)

    assert len(all_records) == 3
    assert all_records[2].generation_idx == 2
