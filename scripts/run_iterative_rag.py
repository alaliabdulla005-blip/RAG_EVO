"""Iterative RAG baseline — 2-round adaptive retrieval on all 50 debug records.

Algorithm:
  Round 1 : BM25 retrieve (top_k=10) → chain-of-thought answer
  Critique : Ask LLM whether the answer is complete; if not, get a
             refined query
  Round 2  : Retrieve with refined query (top_k=10), merge chunks
             (deduplicated by text), generate final answer

Fixed config: chunk_size=384, overlap=0.3, temperature=0.0.

DO NOT run this script until the Phase 12 GA has started — it shares
the SambaNova API quota and should not compete with the GA.

Run from project root:
    python scripts/run_iterative_rag.py

Requires INFERENCE_PROVIDER and matching API key in .env.
"""

import json
import logging
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

from rag_evo.data.hotpotqa_loader import load_records
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
from rag_evo.utils.paths import get_project_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

_CITATION_RE = re.compile(r"\[DOC_\d+\]")

_CRITIQUE_SYSTEM_PROMPT = (
    "You are a review assistant. Given a question and a proposed answer, "
    "determine if the answer is complete and well-supported. If not, "
    "suggest a refined search query that would help find the missing "
    'information. Respond with ONLY the refined query, or "SUFFICIENT" '
    "if the answer is complete."
)

CHUNK_SIZE = 384
OVERLAP = 0.3
TOP_K_PER_ROUND = 10

OUT_PATH = Path("results/ablations/iterative_rag.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

paths = get_project_paths()

print("Loading HotpotQA debug subset...")
debug_records, corpus_documents = load_records(
    Path("data/processed/hotpotqa_debug")
)
print(f"Loaded {len(debug_records)} records, {len(corpus_documents)} corpus docs")

wrapper = GeneratorWrapper()
weights = FitnessWeights()

# Build BM25 index once — reused for both rounds
print("Building BM25 index ...")
retriever_cfg = RetrieverConfig(
    retriever_type="BM25",
    chunk_size=CHUNK_SIZE,
    overlap=OVERLAP,
    top_k=TOP_K_PER_ROUND,
    reranker_enabled=False,
)
retrieve_fn = build_retriever(retriever_cfg, corpus_documents)

decoding_main = DecodingConfig(temperature=0.0, max_tokens=256, seed=42)
decoding_critique = DecodingConfig(temperature=0.0, max_tokens=64, seed=42)

pipeline_results: list[PipelineResult] = []

for i, record in enumerate(debug_records, start=1):
    print(f"Record {i}/{len(debug_records)}: {record.record_id}")

    # ── Round 1: standard retrieval ───────────────────────────────────────────
    r1_result = retrieve_fn(record.question, TOP_K_PER_ROUND)
    r1_chunks = [chunk for chunk, _ in r1_result.retrieved_chunks]
    r1_context = build_context_block(r1_chunks)

    r1_output = wrapper.generate(
        system_prompt=SYSTEM_PROMPT_CHAIN_OF_THOUGHT,
        context_block=r1_context,
        query=record.question,
        decoding=decoding_main,
    )

    # ── Self-critique ─────────────────────────────────────────────────────────
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

    if refined_query.upper() == "SUFFICIENT":
        # Round-1 answer is complete — no second retrieval needed
        final_answer = r1_output.answer
        final_chunks = r1_chunks
        final_input_tokens = (
            r1_output.input_tokens + critique_output.input_tokens
        )
        final_output_tokens = r1_output.output_tokens
        final_latency_ms = r1_output.latency_ms + critique_output.latency_ms
        final_raw_prompt = r1_output.raw_prompt
    else:
        # ── Round 2: retrieve with refined query ──────────────────────────────
        r2_result = retrieve_fn(refined_query, TOP_K_PER_ROUND)
        r2_chunks = [chunk for chunk, _ in r2_result.retrieved_chunks]

        # Merge round-1 and round-2 chunks, deduplicating by text
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
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP,
            top_k=TOP_K_PER_ROUND,
            raw_prompt=final_raw_prompt,
            raw_answer=final_answer,
        )
    )

# ── Evaluate ──────────────────────────────────────────────────────────────────
harness = EvaluationHarness(debug_records)
_, aggregate = harness.evaluate_batch(pipeline_results)
components = compute_fitness_components(aggregate, weights)

print(f"\n{'=' * 60}")
print(f"Iterative RAG: fitness={components.fitness:.4f}")
print(f"  EM={aggregate.mean_em:.4f} F1={aggregate.mean_f1:.4f}")
print(
    f"  CitP={aggregate.mean_citation_precision:.4f} "
    f"CitR={aggregate.mean_citation_recall:.4f} "
    f"CitF1={aggregate.mean_citation_f1:.4f}"
)
print(
    f"  RetR@k={aggregate.mean_retrieval_recall_at_k:.4f} "
    f"Tokens={aggregate.mean_input_tokens:.0f} "
    f"Lat={aggregate.mean_latency_ms:.0f}ms"
)

summary = {
    "system": "iterative_rag",
    "config": {
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "top_k_per_round": TOP_K_PER_ROUND,
        "prompt_strategy": "chain-of-thought",
        "temperature": 0.0,
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
    "n_records": len(debug_records),
}
OUT_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"\nSaved to {OUT_PATH}")

# Force exit — httpx connection pool in the openai/groq client holds
# non-daemon threads that prevent natural process termination.
sys.exit(0)
