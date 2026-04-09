"""Budget checkpoint — measure actual token costs and latency.
Run from project root: python scripts/budget_check.py
"""
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

from rag_evo.cache.cache import ResultCache
from rag_evo.config.schema import MicroPilotConfig
from rag_evo.data.hotpotqa_loader import load_records
from rag_evo.evaluation.harness import EvaluationHarness
from rag_evo.generation.wrapper import GeneratorWrapper
from rag_evo.runner.experiment_runner import (
    ExperimentRunner,
    RunConfig,
    generate_run_id,
)

print("=== Budget Checkpoint — Phase 10 Pre-flight ===\n")

qa_records, corpus_docs = load_records(
    Path("data/processed/hotpotqa_debug")
)

wrapper = GeneratorWrapper()
harness = EvaluationHarness(qa_records=qa_records)
cache = ResultCache(
    Path("data/cache/budget_check"),
    enabled=True
)

config = MicroPilotConfig(
    chunk_size=256,
    overlap=0.0,
    top_k=5,
    prompt_strategy="zero-shot",
    temperature=0.0,
)

run_id = generate_run_id(config)
run_config = RunConfig(
    pipeline_config=config,
    run_id=run_id,
    qa_records=qa_records[:10],
    corpus_documents=corpus_docs,
    cache_dir=Path("data/cache/budget_check"),
    results_dir=Path("results/runs/budget_check"),
    model_id=os.getenv("GROQ_MODEL_ID", "llama-3.1-8b-instant"),
    seed=42,
)

runner = ExperimentRunner(
    wrapper=wrapper,
    harness=harness,
    cache=cache,
)

print("Running 10 records through full pipeline...")
print(f"Config: chunk_size={config.chunk_size}, "
      f"overlap={config.overlap}, "
      f"top_k={config.top_k}, "
      f"strategy={config.prompt_strategy}\n")

t0 = time.perf_counter()
result = runner.run(run_config)
t1 = time.perf_counter()

m = result.aggregate_metrics
wall_time = t1 - t0

print("=== Measured Actuals (n=10) ===")
print(f"Wall time:           {wall_time:.1f}s")
print(f"Cache hits:          {result.cache_hits}")
print(f"Cache misses:        {result.cache_misses}")
print(f"Mean input tokens:   {m.mean_input_tokens:.1f}")
print(f"Mean output tokens:  {m.mean_output_tokens:.1f}")
print(f"Mean latency (ms):   {m.mean_latency_ms:.1f}")
print(f"Total input tokens:  {m.total_input_tokens}")
print(f"Total output tokens: {m.total_output_tokens}")
print(f"Total tokens:        {m.total_input_tokens + m.total_output_tokens}")

# GA pilot projections
pop = 10
gen = 15
records = 50
cache_hit_rate = 0.20
unique_calls = int(pop * gen * records * (1 - cache_hit_rate))
worst_calls = pop * gen * records

mean_tokens = m.mean_input_tokens + m.mean_output_tokens
rpm_limit = 30
tpm_limit = 30000
daily_limit = 700000  # conservative estimate

total_tokens_realistic = unique_calls * mean_tokens
total_tokens_worst = worst_calls * mean_tokens
time_realistic_min = unique_calls / rpm_limit
time_worst_min = worst_calls / rpm_limit
days_realistic = total_tokens_realistic / daily_limit
days_worst = total_tokens_worst / daily_limit

print("\n=== GA Pilot Projections ===")
print(f"Population: {pop}, Generations: {gen}, Records: {records}")
print(f"Assumed cache hit rate: {cache_hit_rate*100:.0f}%")
print("")
print(f"Total API calls (worst):    {worst_calls:,}")
print(f"Total API calls (realistic): {unique_calls:,}")
print("")
print(f"Total tokens (worst):    {total_tokens_worst:,.0f}")
print(f"Total tokens (realistic): {total_tokens_realistic:,.0f}")
print("")
print(f"Active query time (worst):    {time_worst_min:.0f} min")
print(f"Active query time (realistic): {time_realistic_min:.0f} min")
print("")
print(f"Days needed at {daily_limit:,} tokens/day (worst):    {days_worst:.1f}")
print(f"Days needed at {daily_limit:,} tokens/day (realistic): {days_realistic:.1f}")
print("")
per_gen_calls = pop * records
per_gen_tokens = per_gen_calls * mean_tokens
per_gen_time_min = per_gen_calls / rpm_limit
print(f"Per generation: {per_gen_calls} calls, "
      f"{per_gen_tokens:,.0f} tokens, "
      f"~{per_gen_time_min:.0f} min active querying")

print("\n=== PASTE THIS FOR ANALYSIS ===")
print(f"Actuals: input={m.mean_input_tokens:.1f} "
      f"output={m.mean_output_tokens:.1f} "
      f"latency={m.mean_latency_ms:.1f}ms "
      f"wall={wall_time:.1f}s")
print(f"Projections: worst={total_tokens_worst:,.0f} tokens "
      f"realistic={total_tokens_realistic:,.0f} tokens")
print(f"Per generation: {per_gen_calls} calls "
      f"{per_gen_tokens:,.0f} tokens "
      f"{per_gen_time_min:.0f}min")
print(f"Days needed: worst={days_worst:.1f} realistic={days_realistic:.1f}")
print("Budget checkpoint complete.")
