"""Phase 6 verification — evaluation harness on Phase 5 results.
Run from project root: python scripts/verify_phase6.py
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

from rag_evo.data.hotpotqa_loader import load_records
from rag_evo.evaluation.harness import EvaluationHarness
from rag_evo.generation.config import DecodingConfig
from rag_evo.generation.wrapper import GeneratorWrapper
from rag_evo.pipeline.rag_pipeline import RAGPipeline
from rag_evo.retrieval.bm25_retriever import build_and_save_index
from rag_evo.retrieval.chunker import chunk_corpus

print("Loading data...")
qa_records, corpus_docs = load_records(Path("data/processed/hotpotqa_debug"))

print("Building index and pipeline...")
chunks = chunk_corpus(corpus_docs, chunk_size=256, overlap=0.0)
retriever = build_and_save_index(chunks, Path("data/indexes/bm25_debug"))
wrapper = GeneratorWrapper()
pipeline = RAGPipeline(
    retriever=retriever,
    wrapper=wrapper,
    chunk_size=256,
    overlap=0.0,
    top_k=5,
    prompt_strategy="zero-shot",
    decoding=DecodingConfig(temperature=0.0, max_tokens=150, seed=42),
)

print("Running pipeline on first 10 records...")
results = pipeline.run_batch(qa_records[:10], log_every=5)

print("Running evaluation harness...")
harness = EvaluationHarness(qa_records=qa_records)
per_record, aggregate = harness.evaluate_batch(results)

print(f"\n=== Aggregate Metrics (n={aggregate.num_records}) ===")
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

print("\n=== Per-Record EM and Citation Recall ===")
for m in per_record:
    print(
        f"  {m.record_id[:12]}... "
        f"EM={m.em:.1f} "
        f"F1={m.f1:.3f} "
        f"CitR={m.citation_recall:.3f} "
        f"RetR@5={m.retrieval_recall_at_k:.3f}"
    )

harness.save_metrics(
    per_record,
    aggregate,
    Path("results/evaluation/bm25_baseline_eval"),
)
print("\nSaved to results/evaluation/bm25_baseline_eval/")
print("Phase 6 verification complete.")
