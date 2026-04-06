"""Phase 9 verification — caching and experiment runner.
Run from project root: python scripts/verify_phase9.py
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

from rag_evo.cache.cache import ResultCache, make_cache_key
from rag_evo.config.schema import MicroPilotConfig
from rag_evo.data.hotpotqa_loader import load_records
from rag_evo.evaluation.harness import EvaluationHarness
from rag_evo.generation.wrapper import GeneratorWrapper
from rag_evo.runner.experiment_runner import (
    ExperimentRunner,
    RunConfig,
    generate_run_id,
)

print("=== Phase 9 — Caching and Experiment Runner Verification ===\n")

# 1. Load data
print("Loading data...")
qa_records, corpus_docs = load_records(
    Path("data/processed/hotpotqa_debug")
)
print(f"Loaded {len(qa_records)} records, {len(corpus_docs)} docs\n")

# 2. Cache key tests
print("Testing cache keys...")
config = MicroPilotConfig(chunk_size=256, top_k=5)
record = qa_records[0]
model_id = os.getenv("GROQ_MODEL_ID", "llama-3.1-8b-instant")

key1 = make_cache_key(record, config, model_id)
key2 = make_cache_key(record, config, model_id)
key3 = make_cache_key(record, MicroPilotConfig(chunk_size=512), model_id)
key4 = make_cache_key(qa_records[1], config, model_id)

assert key1 == key2, "Same inputs must produce same key"
assert key1 != key3, "Different config must produce different key"
assert key1 != key4, "Different record must produce different key"
print(f"Key (same config):      {key1[:16]}...")
print(f"Key (chunk_size=512):   {key3[:16]}...")
print(f"Key (different record): {key4[:16]}...")
print("Cache key tests: PASSED\n")

# 3. Cache set/get roundtrip
print("Testing cache set/get...")
cache = ResultCache(Path("data/cache/phase9_verify"))
print(f"Initial cache size: {cache.size()}")

# 4. Experiment runner — 3 records, real API calls
print("Running experiment (3 records, BM25, real API)...")
wrapper = GeneratorWrapper()
harness = EvaluationHarness(qa_records=qa_records)
runner = ExperimentRunner(wrapper=wrapper, harness=harness, cache=cache)

config_to_run = MicroPilotConfig(
    chunk_size=256,
    overlap=0.0,
    top_k=5,
    prompt_strategy="zero-shot",
    temperature=0.0,
)
run_id = generate_run_id(config_to_run)
print(f"Run ID: {run_id}")

run_config = RunConfig(
    pipeline_config=config_to_run,
    run_id=run_id,
    qa_records=qa_records[:3],
    corpus_documents=corpus_docs,
    cache_dir=Path("data/cache/phase9_verify"),
    results_dir=Path("results/runs"),
    model_id=model_id,
    seed=42,
)

t0 = time.perf_counter()
result = runner.run(run_config)
t1 = time.perf_counter()
print(f"Run completed in {t1 - t0:.1f}s")

print(f"\nCache hits:   {result.cache_hits}")
print(f"Cache misses: {result.cache_misses}")
print(f"Cache size after run: {cache.size()}")

print(f"\n=== Run Metrics (n={result.aggregate_metrics.num_records}) ===")
m = result.aggregate_metrics
print(f"EM:                 {m.mean_em:.4f}")
print(f"F1:                 {m.mean_f1:.4f}")
print(f"Citation Recall:    {m.mean_citation_recall:.4f}")
print(f"Retrieval Recall@5: {m.mean_retrieval_recall_at_k:.4f}")
print(f"Mean input tokens:  {m.mean_input_tokens:.1f}")
print(f"Mean output tokens: {m.mean_output_tokens:.1f}")
print(f"Mean latency (ms):  {m.mean_latency_ms:.1f}")

# 5. Cache hit test — run same config again
print("\nRunning same config again (should be all cache hits)...")
run_id2 = generate_run_id(config_to_run)
run_config2 = RunConfig(
    pipeline_config=config_to_run,
    run_id=run_id2,
    qa_records=qa_records[:3],
    corpus_documents=corpus_docs,
    cache_dir=Path("data/cache/phase9_verify"),
    results_dir=Path("results/runs"),
    model_id=model_id,
    seed=42,
)
t0 = time.perf_counter()
result2 = runner.run(run_config2)
t1 = time.perf_counter()
print(f"Second run completed in {t1 - t0:.1f}s")
print(f"Cache hits:   {result2.cache_hits}")
print(f"Cache misses: {result2.cache_misses}")

assert result2.cache_hits == 3, "All 3 records should be cache hits"
assert result2.cache_misses == 0, "No misses on second run"
print("Cache hit test: PASSED")

print("\n=== PASTE THIS SUMMARY FOR ANALYSIS ===")
print(
    f"Run 1 — misses={result.cache_misses} hits={result.cache_hits} "
    f"time={result.total_wall_time_s:.1f}s"
)
print(
    f"Run 2 — misses={result2.cache_misses} hits={result2.cache_hits} "
    f"time={result2.total_wall_time_s:.1f}s"
)
print(
    f"Speedup: {result.total_wall_time_s / result2.total_wall_time_s:.1f}x"
)
print(
    f"EM={m.mean_em:.4f} F1={m.mean_f1:.4f} "
    f"CitR={m.mean_citation_recall:.4f} "
    f"RetR@5={m.mean_retrieval_recall_at_k:.4f}"
)
print(f"Cache size: {cache.size()} entries")
print("Phase 9 verification complete.")
