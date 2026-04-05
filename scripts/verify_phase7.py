"""Phase 7 verification — dense, hybrid retrievers and reranker.
Run from project root: python scripts/verify_phase7.py
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

from rag_evo.data.hotpotqa_loader import load_records
from rag_evo.evaluation.harness import EvaluationHarness
from rag_evo.generation.config import DecodingConfig
from rag_evo.generation.wrapper import GeneratorWrapper
from rag_evo.pipeline.rag_pipeline import RAGPipeline
from rag_evo.retrieval.chunker import chunk_corpus
from rag_evo.retrieval.dense_retriever import DenseRetriever
from rag_evo.retrieval.hybrid_retriever import HybridRetriever
from rag_evo.retrieval.retriever_factory import RetrieverConfig, build_retriever

RECORDS_PATH = Path("data/processed/hotpotqa_debug")
N_RECORDS = 5


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _run_retrieval_smoke(
    retriever_callable,
    query: str,
    label: str,
    top_k: int = 3,
) -> None:
    result = retriever_callable(query, top_k)
    print(f"\n[{label}] query='{query}' top_k={top_k}")
    print(f"  retriever_type : {result.retriever_type}")
    print(f"  num results    : {len(result.retrieved_chunks)}")
    for i, (chunk, score) in enumerate(result.retrieved_chunks):
        print(f"  [{i + 1}] doc_id={chunk.doc_id!r} score={score:.4f}")
        print(f"       {chunk.text[:80]!r}")


print("Loading data...")
qa_records, corpus_docs = load_records(RECORDS_PATH)
print(f"  {len(qa_records)} QA records, {len(corpus_docs)} corpus documents")

print("Chunking corpus (chunk_size=256, overlap=0.0)...")
chunks = chunk_corpus(corpus_docs, chunk_size=256, overlap=0.0)
print(f"  {len(chunks)} chunks produced")

sample_query = qa_records[0].question
print(f"\nSample query: {sample_query!r}")

# ── Dense Retriever ────────────────────────────────────────────────────────────
_section("Dense Retriever (SentenceTransformer + FAISS)")
print("  Building dense index (this may take a moment)...")
dense = DenseRetriever(chunks)
_run_retrieval_smoke(dense.retrieve, sample_query, "dense")

# ── Hybrid Retriever ───────────────────────────────────────────────────────────
_section("Hybrid Retriever (alpha=0.5)")
print("  Building hybrid index...")
hybrid = HybridRetriever(chunks, alpha=0.5)
_run_retrieval_smoke(hybrid.retrieve, sample_query, "hybrid")

# ── RetrieverFactory — BM25 + reranker disabled ────────────────────────────────
_section("RetrieverFactory — BM25 (no reranker)")
config_bm25 = RetrieverConfig(
    retriever_type="BM25",
    chunk_size=256,
    overlap=0.0,
    top_k=3,
    reranker_enabled=False,
)
retrieve_bm25 = build_retriever(config_bm25, corpus_docs)
_run_retrieval_smoke(retrieve_bm25, sample_query, "factory/BM25")

# ── Pipeline with dense retriever ─────────────────────────────────────────────
_section(f"End-to-end pipeline (dense, {N_RECORDS} records)")
wrapper = GeneratorWrapper()

pipeline = RAGPipeline(
    retriever=dense,
    wrapper=wrapper,
    chunk_size=256,
    overlap=0.0,
    top_k=3,
    prompt_strategy="zero-shot",
    decoding=DecodingConfig(temperature=0.0, max_tokens=150, seed=42),
)

print(f"  Running pipeline on first {N_RECORDS} records...")
results = pipeline.run_batch(qa_records[:N_RECORDS], log_every=1)

harness = EvaluationHarness(qa_records=qa_records)
per_record, aggregate = harness.evaluate_batch(results)

print(f"\n=== Aggregate Metrics (dense, n={aggregate.num_records}) ===")
print(f"EM:                  {aggregate.mean_em:.4f}")
print(f"F1:                  {aggregate.mean_f1:.4f}")
print(f"Citation Precision:  {aggregate.mean_citation_precision:.4f}")
print(f"Citation Recall:     {aggregate.mean_citation_recall:.4f}")
print(f"Citation F1:         {aggregate.mean_citation_f1:.4f}")
print(f"Retrieval Recall@3:  {aggregate.mean_retrieval_recall_at_k:.4f}")
print(f"Mean latency (ms):   {aggregate.mean_latency_ms:.1f}")

print("\n=== Per-Record Summary ===")
for m in per_record:
    print(
        f"  {m.record_id[:12]}... "
        f"EM={m.em:.1f} "
        f"F1={m.f1:.3f} "
        f"CitR={m.citation_recall:.3f} "
        f"RetR@3={m.retrieval_recall_at_k:.3f}"
    )

out_dir = Path("results/evaluation/dense_phase7_sample")
harness.save_metrics(per_record, aggregate, out_dir)
print(f"\nSaved to {out_dir}/")

print("\n" + "=" * 60)
print("  PASTE THIS SUMMARY")
print("=" * 60)
print(f"Phase 7 verify — dense retriever, {N_RECORDS} records")
print(f"  Chunks: {len(chunks)}")
print(f"  EM={aggregate.mean_em:.4f}  F1={aggregate.mean_f1:.4f}")
print(
    f"  CitP={aggregate.mean_citation_precision:.4f}  "
    f"CitR={aggregate.mean_citation_recall:.4f}  "
    f"CitF1={aggregate.mean_citation_f1:.4f}"
)
print(f"  RetR@3={aggregate.mean_retrieval_recall_at_k:.4f}")
print(f"  Mean latency: {aggregate.mean_latency_ms:.1f} ms")
print("Phase 7 verification complete.")
