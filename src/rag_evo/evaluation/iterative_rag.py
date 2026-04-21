"""2-round adaptive iterative RAG evaluation function.

Extracted from scripts/run_iterative_rag.py and scripts/run_iterative_rag_legal.py
to eliminate duplication and support reuse across development and held-out evaluation.
"""

import logging
import re
from pathlib import Path

from rag_evo.data.schemas import CorpusDocument, QARecord
from rag_evo.evaluation.harness import EvaluationHarness
from rag_evo.ga.fitness import FitnessWeights, compute_fitness_components
from rag_evo.generation.config import DecodingConfig
from rag_evo.generation.wrapper import GeneratorWrapper
from rag_evo.pipeline.prompt_builder import (
    SYSTEM_PROMPT_CHAIN_OF_THOUGHT,
    build_context_block,
)
from rag_evo.pipeline.rag_pipeline import PipelineResult
from rag_evo.retrieval.retriever_factory import RetrieverConfig, build_retriever

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[DOC_\d+\]")

_CRITIQUE_SYSTEM_PROMPT = (
    "You are a review assistant. Given a question and a proposed answer, "
    "determine if the answer is complete and well-supported. If not, "
    "suggest a refined search query that would help find the missing "
    'information. Respond with ONLY the refined query, or "SUFFICIENT" '
    "if the answer is complete."
)


def run_iterative_rag(
    qa_records: list[QARecord],
    corpus_documents: list[CorpusDocument],
    config: dict,
    wrapper: GeneratorWrapper,
    harness: EvaluationHarness,
    cache_dir: Path,  # noqa: ARG001 — accepted for interface consistency, unused in v1
    fitness_weights: FitnessWeights,
) -> dict:
    """Run 2-round adaptive retrieval evaluation on a list of records.

    Algorithm:
      Round 1 : BM25 retrieve (top_k_per_round) → chain-of-thought answer.
      Critique : Ask LLM if the answer is complete; if not, get a refined query.
      Round 2  : Retrieve with refined query, merge chunks (dedup by text),
                 generate final answer.

    Args:
        qa_records: Records to evaluate. The EvaluationHarness must be bound
            to the same set.
        corpus_documents: Full corpus used to build the BM25 index.
        config: Hyperparameter overrides. Accepted keys and defaults:
            ``chunk_size`` (int, 384), ``overlap`` (float, 0.3),
            ``top_k_per_round`` (int, 10), ``temperature`` (float, 0.0),
            ``max_tokens`` (int, 256), ``seed`` (int, 42).
        wrapper: Shared GeneratorWrapper instance.
        harness: EvaluationHarness bound to *qa_records*.
        cache_dir: Reserved for future result caching; unused in v1.
        fitness_weights: Weights for scalar fitness computation.

    Returns:
        Summary dict with keys: ``system``, ``config``, ``metrics``,
        ``fitness``, ``n_records``, ``n_round2_triggered``, ``per_record``.
    """
    chunk_size: int = config.get("chunk_size", 384)
    overlap: float = config.get("overlap", 0.3)
    top_k: int = config.get("top_k_per_round", 10)
    temperature: float = config.get("temperature", 0.0)
    max_tokens: int = config.get("max_tokens", 256)
    seed: int = config.get("seed", 42)

    n = len(qa_records)
    logger.info(
        "Building BM25 index (chunk_size=%d, overlap=%.1f)...", chunk_size, overlap
    )
    retriever_cfg = RetrieverConfig(
        retriever_type="BM25",
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k,
        reranker_enabled=False,
    )
    retrieve_fn = build_retriever(retriever_cfg, corpus_documents)

    decoding_main = DecodingConfig(
        temperature=temperature, max_tokens=max_tokens, seed=seed
    )
    decoding_critique = DecodingConfig(temperature=0.0, max_tokens=64, seed=seed)

    pipeline_results: list[PipelineResult] = []
    per_record_details: list[dict] = []

    for i, record in enumerate(qa_records, start=1):
        logger.info("Record %d/%d: %s", i, n, record.record_id)

        # ── Round 1: standard retrieval ───────────────────────────────────────
        r1_result = retrieve_fn(record.question, top_k)
        r1_chunks = [chunk for chunk, _ in r1_result.retrieved_chunks]
        r1_context = build_context_block(r1_chunks)

        r1_output = wrapper.generate(
            system_prompt=SYSTEM_PROMPT_CHAIN_OF_THOUGHT,
            context_block=r1_context,
            query=record.question,
            decoding=decoding_main,
        )

        # ── Self-critique ─────────────────────────────────────────────────────
        critique_context = (
            f"Question: {record.question}\n\nAnswer: {r1_output.answer}"
        )
        critique_output = wrapper.generate(
            system_prompt=_CRITIQUE_SYSTEM_PROMPT,
            context_block=critique_context,
            query="Is the answer complete?",
            decoding=decoding_critique,
        )
        refined_query = critique_output.answer.strip()
        round_used = 1

        if refined_query.upper() == "SUFFICIENT":
            final_answer = r1_output.answer
            final_chunks = r1_chunks
            final_input_tokens = r1_output.input_tokens + critique_output.input_tokens
            final_output_tokens = r1_output.output_tokens
            final_latency_ms = r1_output.latency_ms + critique_output.latency_ms
            final_raw_prompt = r1_output.raw_prompt
        else:
            # ── Round 2: retrieve with refined query ──────────────────────────
            round_used = 2
            r2_result = retrieve_fn(refined_query, top_k)
            r2_chunks = [chunk for chunk, _ in r2_result.retrieved_chunks]

            seen_texts: set[str] = {c.text for c in r1_chunks}
            merged_chunks = list(r1_chunks)
            for c in r2_chunks:
                if c.text not in seen_texts:
                    merged_chunks.append(c)
                    seen_texts.add(c.text)

            merged_context = build_context_block(merged_chunks)
            r2_output = wrapper.generate(
                system_prompt=SYSTEM_PROMPT_CHAIN_OF_THOUGHT,
                context_block=merged_context,
                query=record.question,
                decoding=decoding_main,
            )

            final_answer = r2_output.answer
            final_chunks = merged_chunks
            final_input_tokens = (
                r1_output.input_tokens
                + critique_output.input_tokens
                + r2_output.input_tokens
            )
            final_output_tokens = r2_output.output_tokens
            final_latency_ms = (
                r1_output.latency_ms
                + critique_output.latency_ms
                + r2_output.latency_ms
            )
            final_raw_prompt = r2_output.raw_prompt

        cited_chunk_ids = _CITATION_RE.findall(final_answer)

        pipeline_results.append(
            PipelineResult(
                record_id=record.record_id,
                question=record.question,
                answer=final_answer,
                cited_chunk_ids=cited_chunk_ids,
                retrieved_chunks=final_chunks,
                input_tokens=final_input_tokens,
                output_tokens=final_output_tokens,
                latency_ms=final_latency_ms,
                retriever_type="bm25",
                prompt_strategy="chain-of-thought",
                chunk_size=chunk_size,
                overlap=overlap,
                top_k=top_k,
                raw_prompt=final_raw_prompt,
                raw_answer=final_answer,
            )
        )

        per_record_details.append({
            "record_id": record.record_id,
            "question": record.question,
            "gold_answer": record.gold_answer,
            "round_used": round_used,
            "critique_result": refined_query,
            "final_answer": final_answer,
            "cited_chunk_ids": cited_chunk_ids,
            "n_chunks_round1": len(r1_chunks),
            "n_chunks_merged": len(final_chunks),
            "input_tokens": final_input_tokens,
            "output_tokens": final_output_tokens,
            "latency_ms": final_latency_ms,
        })

    # ── Evaluate ──────────────────────────────────────────────────────────────
    per_record_metrics, aggregate = harness.evaluate_batch(pipeline_results)
    components = compute_fitness_components(aggregate, fitness_weights)

    n_round2 = sum(1 for d in per_record_details if d["round_used"] == 2)
    logger.info(
        "fitness=%.4f F1=%.4f CitF1=%.4f RetR@k=%.4f Round2=%d/%d",
        components.fitness,
        aggregate.mean_f1,
        aggregate.mean_citation_f1,
        aggregate.mean_retrieval_recall_at_k,
        n_round2,
        n,
    )

    return {
        "system": "iterative_rag",
        "config": {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "top_k_per_round": top_k,
            "prompt_strategy": "chain-of-thought",
            "temperature": temperature,
            "rounds": 2,
        },
        "metrics": {
            "mean_em": aggregate.mean_em,
            "mean_f1": aggregate.mean_f1,
            "mean_citation_precision": aggregate.mean_citation_precision,
            "mean_citation_recall": aggregate.mean_citation_recall,
            "mean_citation_f1": aggregate.mean_citation_f1,
            "mean_retrieval_recall_at_k": aggregate.mean_retrieval_recall_at_k,
            "mean_input_tokens": aggregate.mean_input_tokens,
            "mean_latency_ms": aggregate.mean_latency_ms,
        },
        "fitness": components.fitness,
        "components": {
            "q_norm": components.q_norm,
            "g_norm": components.g_norm,
            "r_norm": components.r_norm,
            "c_norm": components.c_norm,
            "l_norm": components.l_norm,
        },
        "n_records": n,
        "n_round2_triggered": n_round2,
        "per_record": per_record_details,
        "per_record_metrics": per_record_metrics,
        "predictions": [d["final_answer"] for d in per_record_details],
    }
