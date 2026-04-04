"""Phase 5 verification — end-to-end RAG baseline on debug subset.
Run from project root: python scripts/verify_phase5.py
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

from rag_evo.data.hotpotqa_loader import load_records
from rag_evo.generation.config import DecodingConfig
from rag_evo.generation.wrapper import GeneratorWrapper
from rag_evo.pipeline.rag_pipeline import RAGPipeline, save_results
from rag_evo.retrieval.bm25_retriever import build_and_save_index
from rag_evo.retrieval.chunker import chunk_corpus

print("Loading data...")
qa_records, corpus_docs = load_records(Path("data/processed/hotpotqa_debug"))

print("Building index...")
chunks = chunk_corpus(corpus_docs, chunk_size=256, overlap=0.0)
retriever = build_and_save_index(chunks, Path("data/indexes/bm25_debug"))

print("Initialising pipeline...")
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

print("Running pipeline on first 5 records...")
results = pipeline.run_batch(qa_records[:5], log_every=1)

print(f"\nProcessed {len(results)} records")
for r in results:
    print(f"\nQ: {r.question}")
    print(f"A: {r.answer}")
    print(f"Cited: {r.cited_chunk_ids}")
    print(f"Tokens: {r.input_tokens} in / {r.output_tokens} out")
    print(f"Latency: {r.latency_ms:.1f}ms")

save_results(results, Path("results/baseline/bm25_baseline_sample.jsonl"))
print("\nSaved to results/baseline/bm25_baseline_sample.jsonl")
print("Phase 5 verification complete.")
