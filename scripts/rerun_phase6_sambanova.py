"""Re-run Phase 6 evaluation baseline using SambaNova full-precision inference.

This script produces the official baseline metrics for the dissertation.
Development metrics from Groq (Phases 1-9) are superseded by these results.
Run from project root: python scripts/rerun_phase6_sambanova.py
"""
import os
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(".env"))

from openai import OpenAI
from rag_evo.data.hotpotqa_loader import load_records
from rag_evo.retrieval.chunker import chunk_corpus
from rag_evo.retrieval.bm25_retriever import build_and_save_index
from rag_evo.generation.config import DecodingConfig
from rag_evo.pipeline.prompt_builder import build_prompt_components
from rag_evo.pipeline.rag_pipeline import PipelineResult, save_results
from rag_evo.evaluation.harness import EvaluationHarness
from rag_evo.utils.seed import set_global_seed
import re
import dataclasses

set_global_seed(42)

# ── SambaNova client (OpenAI-compatible) ──────────────────
client = OpenAI(
    api_key=os.getenv("SAMBANOVA_API_KEY"),
    base_url=os.getenv("SAMBANOVA_BASE_URL"),
)
MODEL_ID = os.getenv("SAMBANOVA_MODEL_ID", "Meta-Llama-3.1-8B-Instruct")
DECODING = DecodingConfig(temperature=0.0, max_tokens=150, seed=42)

print("=== Phase 6 Baseline Re-run — SambaNova (Full BF16/FP32) ===\n")
print(f"Provider: SambaNova Cloud")
print(f"Model:    {MODEL_ID}")
print(f"Precision: Full BF16/FP32 mixed\n")

# ── Load data ─────────────────────────────────────────────
print("Loading data...")
qa_records, corpus_docs = load_records(Path("data/processed/hotpotqa_debug"))
print(f"Loaded {len(qa_records)} QA records, {len(corpus_docs)} corpus docs\n")

# ── Build index ───────────────────────────────────────────
print("Building BM25 index...")
chunks = chunk_corpus(corpus_docs, chunk_size=256, overlap=0.0)
retriever = build_and_save_index(chunks, Path("data/indexes/bm25_debug"))
print(f"Index built: {len(chunks)} chunks\n")

# ── Run pipeline on first 10 records ─────────────────────
print("Running pipeline on first 10 records...")
results = []

for i, record in enumerate(qa_records[:10]):
    retrieval_result = retriever.retrieve(record.question, top_k=5)
    retrieved_chunks = [c for c, _ in retrieval_result.retrieved_chunks]

    system_prompt, context_block = build_prompt_components(
        retrieved_chunks, record.question, strategy="zero-shot"
    )

    user_message = f"Context:\n{context_block}\n\nQuestion: {record.question}"

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=DECODING.temperature,
        max_tokens=DECODING.max_tokens,
        seed=DECODING.seed,
        top_p=DECODING.top_p,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    answer = response.choices[0].message.content.strip()
    input_tokens  = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cited = re.findall(r'\[DOC_\d+\]', answer)

    result = PipelineResult(
        record_id=record.record_id,
        question=record.question,
        answer=answer,
        cited_chunk_ids=cited,
        retrieved_chunks=retrieved_chunks,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        retriever_type="bm25",
        prompt_strategy="zero-shot",
        chunk_size=256,
        overlap=0.0,
        top_k=5,
        raw_prompt=user_message,
        raw_answer=answer,
    )
    results.append(result)
    print(f"  [{i+1:02d}] {record.question[:60]}...")
    print(f"        → {answer[:80]}")
    print(f"        tokens: {input_tokens}in/{output_tokens}out  "
          f"latency: {latency_ms:.0f}ms  cited: {cited}")

# ── Evaluate ──────────────────────────────────────────────
print("\nRunning evaluation harness...")
harness = EvaluationHarness(qa_records=qa_records)
per_record, aggregate = harness.evaluate_batch(results)

# ── Print results ─────────────────────────────────────────
print(f"\n=== Official Baseline Metrics — SambaNova (n={aggregate.num_records}) ===")
print(f"Provider:            SambaNova Cloud (full BF16/FP32)")
print(f"Model:               {MODEL_ID}")
print(f"EM:                  {aggregate.mean_em:.4f}")
print(f"F1:                  {aggregate.mean_f1:.4f}")
print(f"Citation Precision:  {aggregate.mean_citation_precision:.4f}")
print(f"Citation Recall:     {aggregate.mean_citation_recall:.4f}")
print(f"Citation F1:         {aggregate.mean_citation_f1:.4f}")
print(f"Retrieval Recall@5:  {aggregate.mean_retrieval_recall_at_k:.4f}")
print(f"Mean input tokens:   {aggregate.mean_input_tokens:.1f}")
print(f"Mean output tokens:  {aggregate.mean_output_tokens:.1f}")
print(f"Mean latency (ms):   {aggregate.mean_latency_ms:.1f}")
print(f"Total input tokens:  {aggregate.total_input_tokens}")
print(f"Total output tokens: {aggregate.total_output_tokens}")

print("\n=== Per-Record Summary ===")
for m in per_record:
    print(f"  {m.record_id[:12]}...  "
          f"EM={m.em:.1f}  F1={m.f1:.3f}  "
          f"CitR={m.citation_recall:.3f}  RetR@5={m.retrieval_recall_at_k:.3f}")

# ── Save official results ─────────────────────────────────
out_dir = Path("results/evaluation/bm25_baseline_eval")
harness.save_metrics(per_record, aggregate, out_dir)
save_results(results, out_dir / "pipeline_results.jsonl")
print(f"\nOfficial baseline saved to {out_dir}/")
print("NOTE: These results supersede Groq development metrics from Phases 1-9.")

print("\n=== PASTE THIS SUMMARY FOR ANALYSIS ===")
print(f"Provider: SambaNova | Model: {MODEL_ID} | Precision: Full BF16/FP32")
print(f"EM={aggregate.mean_em:.4f}  F1={aggregate.mean_f1:.4f}")
print(f"CitP={aggregate.mean_citation_precision:.4f}  "
      f"CitR={aggregate.mean_citation_recall:.4f}  "
      f"CitF1={aggregate.mean_citation_f1:.4f}")
print(f"RetR@5={aggregate.mean_retrieval_recall_at_k:.4f}")
print(f"Mean tokens: {aggregate.mean_input_tokens:.1f}in / "
      f"{aggregate.mean_output_tokens:.1f}out")
print(f"Mean latency: {aggregate.mean_latency_ms:.1f}ms")
print(f"Total tokens: {aggregate.total_input_tokens}in / "
      f"{aggregate.total_output_tokens}out")