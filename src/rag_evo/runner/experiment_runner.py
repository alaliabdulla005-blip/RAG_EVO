"""Experiment runner for RAG-Evo GA evaluation loop.

Wraps the full pipeline (chunking → indexing → retrieval →
generation → evaluation) into a single callable that the GA
can use to evaluate any configuration.

The runner is the innermost loop of the GA. It must be:
- Fast: use caching to avoid redundant API calls
- Resilient: continue on individual record failures
- Logged: save every run's artifacts to disk
- Reproducible: deterministic given the same config and seed
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from rag_evo.cache.cache import ResultCache, make_cache_key
from rag_evo.config.schema import FullV1Config, MicroPilotConfig
from rag_evo.evaluation.harness import AggregateMetrics, EvaluationHarness
from rag_evo.generation.config import DecodingConfig
from rag_evo.generation.wrapper import GeneratorWrapper
from rag_evo.pipeline.rag_pipeline import PipelineResult, RAGPipeline
from rag_evo.retrieval.reranker import Reranker
from rag_evo.retrieval.retriever_factory import RetrieverConfig, build_retriever

logger = logging.getLogger(__name__)


def generate_run_id(config: MicroPilotConfig | FullV1Config) -> str:
    """Generate a unique run ID incorporating config signature.

    Format: {YYYYMMDD}_{HHMMSS}_{config_hash_8chars}

    Args:
        config: The pipeline config for this run.

    Returns:
        Unique run ID string.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = {
        k: v for k, v in asdict(config).items()
        if k not in {"citations_required", "top_p"}
    }
    config_hash = hashlib.sha256(
        json.dumps(d, sort_keys=True).encode()
    ).hexdigest()[:8]
    return f"{ts}_{config_hash}"


class _RetrievableAdapter:
    """Adapts a retrieve-callable to the .retrieve() interface RAGPipeline expects."""

    def __init__(self, fn) -> None:  # type: ignore[type-arg]
        self._fn = fn

    def retrieve(self, query: str, top_k: int):  # type: ignore[return]
        return self._fn(query, top_k)


@dataclass(frozen=True)
class RunConfig:
    """Complete configuration for a single experiment run.

    Combines the GA chromosome config with runtime settings
    that are fixed across the GA run.

    Args:
        pipeline_config: MicroPilotConfig or FullV1Config.
        run_id: Unique identifier for this run.
        qa_records: List of QARecord objects to evaluate on.
        corpus_documents: List of CorpusDocument objects.
        cache_dir: Directory for result caching.
        results_dir: Directory to save run artifacts.
        model_id: Model identifier for cache keying.
        seed: Random seed for reproducibility.
    """

    pipeline_config: MicroPilotConfig | FullV1Config
    run_id: str
    qa_records: list
    corpus_documents: list
    cache_dir: Path
    results_dir: Path
    model_id: str
    seed: int = 42


@dataclass(frozen=True)
class RunResult:
    """Complete result of one experiment run.

    Args:
        run_id: Matches RunConfig.run_id.
        pipeline_config: The config that was evaluated.
        aggregate_metrics: AggregateMetrics for this run.
        per_record_metrics: List of RecordMetrics.
        cache_hits: Number of cache hits during this run.
        cache_misses: Number of cache misses.
        total_wall_time_s: Total wall clock time in seconds.
        timestamp: ISO format timestamp when run completed.
    """

    run_id: str
    pipeline_config: MicroPilotConfig | FullV1Config
    aggregate_metrics: AggregateMetrics
    per_record_metrics: list
    cache_hits: int
    cache_misses: int
    total_wall_time_s: float
    timestamp: str


class ExperimentRunner:
    """Evaluates a pipeline configuration on a QA record set.

    Args:
        wrapper: GeneratorWrapper instance (shared across runs).
        harness: EvaluationHarness instance (shared across runs).
        cache: ResultCache instance (shared across runs).
        reranker: Optional Reranker instance. Shared if provided.
    """

    def __init__(
        self,
        wrapper: GeneratorWrapper,
        harness: EvaluationHarness,
        cache: ResultCache,
        reranker: Reranker | None = None,
    ) -> None:
        self._wrapper = wrapper
        self._harness = harness
        self._cache = cache
        self._reranker = reranker

    def run(self, run_config: RunConfig) -> RunResult:
        """Evaluate one pipeline configuration on all QA records.

        Args:
            run_config: Complete run configuration.

        Returns:
            RunResult with all metrics and metadata populated.
        """
        start_time = time.perf_counter()
        pc = run_config.pipeline_config

        retriever_type = getattr(pc, "retriever_type", "BM25")
        reranker_enabled = getattr(pc, "reranker_enabled", False)
        max_output_tokens = getattr(pc, "max_output_tokens", 256)

        retriever_cfg = RetrieverConfig(
            retriever_type=retriever_type,
            chunk_size=pc.chunk_size,
            overlap=pc.overlap,
            top_k=pc.top_k,
            reranker_enabled=reranker_enabled,
        )

        retrieve_fn = build_retriever(
            retriever_cfg,
            run_config.corpus_documents,
            reranker=self._reranker if reranker_enabled else None,
        )
        adapted_retriever = _RetrievableAdapter(retrieve_fn)

        decoding = DecodingConfig(
            temperature=pc.temperature,
            max_tokens=max_output_tokens,
            seed=run_config.seed,
        )

        pipeline = RAGPipeline(
            retriever=adapted_retriever,
            wrapper=self._wrapper,
            chunk_size=pc.chunk_size,
            overlap=pc.overlap,
            top_k=pc.top_k,
            prompt_strategy=pc.prompt_strategy,
            decoding=decoding,
        )

        pipeline_results: list[PipelineResult] = []
        cache_hits = 0
        cache_misses = 0

        for record in run_config.qa_records:
            cache_key = make_cache_key(record, pc, run_config.model_id)
            cached = self._cache.get(cache_key)
            if cached is not None:
                pipeline_results.append(cached)
                cache_hits += 1
            else:
                try:
                    result = pipeline.run(record)
                    self._cache.set(cache_key, result)
                    pipeline_results.append(result)
                    cache_misses += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Skipping record %s: %s", record.record_id, exc
                    )

        per_record, aggregate = self._harness.evaluate_batch(pipeline_results)

        elapsed = time.perf_counter() - start_time
        timestamp = datetime.now().isoformat()

        run_result = RunResult(
            run_id=run_config.run_id,
            pipeline_config=pc,
            aggregate_metrics=aggregate,
            per_record_metrics=list(per_record),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            total_wall_time_s=elapsed,
            timestamp=timestamp,
        )

        self._save_run_artifacts(run_config, run_result, pipeline_results)
        logger.info(
            "Run %s complete: hits=%d misses=%d wall=%.1fs",
            run_config.run_id, cache_hits, cache_misses, elapsed,
        )
        return run_result

    def _save_run_artifacts(
        self,
        run_config: RunConfig,
        run_result: RunResult,
        pipeline_results: list[PipelineResult],
    ) -> None:
        """Save all artifacts for a completed run.

        Creates run_config.results_dir / run_config.run_id / with:
        - config.json: The pipeline_config as JSON
        - aggregate_metrics.json: AggregateMetrics
        - per_record_metrics.jsonl: One RecordMetrics per line
        - pipeline_results.jsonl: One PipelineResult per line
        - run_summary.json: RunResult metadata (no nested lists)
        """
        run_dir = run_config.results_dir / run_config.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "config.json").write_text(
            json.dumps(asdict(run_config.pipeline_config), indent=2) + "\n",
            encoding="utf-8",
        )

        (run_dir / "aggregate_metrics.json").write_text(
            json.dumps(asdict(run_result.aggregate_metrics), indent=2) + "\n",
            encoding="utf-8",
        )

        with (run_dir / "per_record_metrics.jsonl").open(
            "w", encoding="utf-8"
        ) as f:
            for m in run_result.per_record_metrics:
                f.write(json.dumps(asdict(m)) + "\n")

        with (run_dir / "pipeline_results.jsonl").open(
            "w", encoding="utf-8"
        ) as f:
            for r in pipeline_results:
                f.write(json.dumps(asdict(r)) + "\n")

        summary = {
            "run_id": run_result.run_id,
            "cache_hits": run_result.cache_hits,
            "cache_misses": run_result.cache_misses,
            "total_wall_time_s": run_result.total_wall_time_s,
            "timestamp": run_result.timestamp,
        }
        (run_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
