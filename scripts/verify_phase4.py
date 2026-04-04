from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

from rag_evo.data.hotpotqa_loader import load_records
from rag_evo.retrieval.bm25_retriever import build_and_save_index, load_index
from rag_evo.retrieval.chunker import chunk_corpus

print("Loading processed records...")
qa_records, corpus_docs = load_records(Path("data/processed/hotpotqa_debug"))
print(f"Loaded {len(qa_records)} QA records and {len(corpus_docs)} docs")

print("Chunking corpus (chunk_size=256, overlap=0.0)...")
chunks = chunk_corpus(corpus_docs, chunk_size=256, overlap=0.0)
print(f"Total chunks produced: {len(chunks)}")

print("Building BM25 index...")
retriever = build_and_save_index(chunks, Path("data/indexes/bm25_debug"))
print("Index saved.")

print("Loading index from disk...")
retriever2 = load_index(Path("data/indexes/bm25_debug"))
print("Index loaded.")

q = qa_records[0].question
print(f"\nQuery: {q}")
result = retriever2.retrieve(q, top_k=5)
print("Top-5 retrieved chunks:")
for i, (chunk, score) in enumerate(result.retrieved_chunks):
    print(f"  [{i + 1}] doc_id={chunk.doc_id} score={score:.4f}")
    print(f"       text={chunk.text[:80]}...")

gold_ids = qa_records[0].source_doc_ids
print(f"\nGold doc IDs: {gold_ids}")
retrieved_doc_ids = [c.doc_id for c, _ in result.retrieved_chunks]
hits = [d for d in gold_ids if d in retrieved_doc_ids]
print(f"Gold docs found in top-5: {hits}")
