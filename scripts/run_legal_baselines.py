"""Legal baseline comparison — four systems on 50 CUAD records.

Systems compared:
  1. Direct LLM         — no retrieval, zero-shot, temperature=0.0
  2. Naive RAG          — BM25, chunk=256, overlap=0, top_k=5, zero-shot
  3. Conservative RAG   — BM25, chunk=256, overlap=0, top_k=3, zero-shot
  4. GA-optimised legal — BM25, chunk=384, overlap=0.2, top_k=20,
                          chain-of-thought, temp=1.0 (Phase 13 best)

All four use legal fitness weights (w_g=0.40 for grounding priority).
Configs 2-4 share one ExperimentRunner.
Config 1 calls the wrapper directly and builds PipelineResult objects.

Run from project root:
    INFERENCE_PROVIDER=sambanova python scripts/run_legal_baselines.py

Requires INFERENCE_PROVIDER and matching API key in .env.
DO NOT run this script — run it manually after Phase 13 GA completes.
"""

import json
import logging
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

from rag_evo.cache.cache import ResultCache
from rag_evo.config.schema import MicroPilotConfig
from rag_evo.data.legal_loader import load_legal_records
from rag_evo.evaluation.harness import EvaluationHarness
from rag_evo.ga.fitness import compute_fitness_components, legal_fitness_weights
from rag_evo.generation.config import DecodingConfig
from rag_evo.generation.wrapper import GeneratorWrapper
from rag_evo.pipeline.prompt_builder import SYSTEM_PROMPT_ZERO_SHOT
from rag_evo.pipeline.rag_pipeline import PipelineResult
from rag_evo.runner.experiment_runner import ExperimentRunner, RunConfig
from rag_evo.utils.paths import get_project_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

_CITATION_RE = re.compile(r"\[DOC_\d+\]")

paths = get_project_paths()

CACHE_DIR = paths.root / "data" / "cache" / "legal_baselines"
OUT_PATH = Path("results/phase14/legal_baselines.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Loading legal debug subset...")
legal_records, corpus_documents = load_legal_records(
    Path("data/processed/legal_debug")
)
print(f"Using {len(legal_records)} records, {len(corpus_documents)} corpus docs")

wrapper = GeneratorWrapper()
harness = EvaluationHarness(legal_records)
cache = ResultCache(CACHE_DIR)
runner = ExperimentRunner(
    wrapper=wrapper, harness=harness, cache=cache
)
weights = legal_fitness_weights()

systems = []

# ── Config 1: Direct LLM (no retrieval) ──────────────────────────────────────
print("\n" + "=" * 50)
print("Config 1: Direct LLM (no retrieval)")
direct_decoding = DecodingConfig(temperature=0.0, max_tokens=150, seed=42)
direct_results: list[PipelineResult] = []

for record in legal_records:
    output = wrapper.generate(
        system_prompt=SYSTEM_PROMPT_ZERO_SHOT,
        context_block="[No context provided]",
        query=record.question,
        decoding=direct_decoding,
    )
    cited = _CITATION_RE.findall(output.answer)
    direct_results.append(
        PipelineResult(
            record_id=record.record_id,
            question=record.question,
            answer=output.answer,
            cited_chunk_ids=cited,
            retrieved_chunks=[],
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
            latency_ms=output.latency_ms,
            retriever_type="none",
            prompt_strategy="zero-shot",
            chunk_size=0,
            overlap=0.0,
            top_k=0,
            raw_prompt=output.raw_prompt,
            raw_answer=output.raw_answer,
        )
    )

_, direct_agg = harness.evaluate_batch(direct_results)
direct_comp = compute_fitness_components(direct_agg, weights)
print(f"  fitness={direct_comp.fitness:.4f} F1={direct_agg.mean_f1:.4f}")

systems.append({
    "name": "Direct LLM",
    "config": "no retrieval, zero-shot, temperature=0.0",
    "metrics": {
        "mean_em": direct_agg.mean_em,
        "mean_f1": direct_agg.mean_f1,
        "mean_citation_precision": direct_agg.mean_citation_precision,
        "mean_citation_recall": direct_agg.mean_citation_recall,
        "mean_citation_f1": direct_agg.mean_citation_f1,
        "mean_retrieval_recall_at_k": direct_agg.mean_retrieval_recall_at_k,
        "mean_input_tokens": direct_agg.mean_input_tokens,
        "mean_latency_ms": direct_agg.mean_latency_ms,
    },
    "fitness": direct_comp.fitness,
})

# ── Configs 2–4: via ExperimentRunner ────────────────────────────────────────
runner_configs = [
    (
        "Naive RAG",
        "BM25, chunk=256, overlap=0, top_k=5, zero-shot, temp=0.0",
        MicroPilotConfig(
            chunk_size=256, overlap=0.0, top_k=5,
            prompt_strategy="zero-shot", temperature=0.0,
        ),
        "naive_rag_legal",
    ),
    (
        "Conservative manual RAG",
        "BM25, chunk=256, overlap=0, top_k=3, zero-shot, temp=0.0",
        MicroPilotConfig(
            chunk_size=256, overlap=0.0, top_k=3,
            prompt_strategy="zero-shot", temperature=0.0,
        ),
        "conservative_rag_legal",
    ),
    (
        "GA-optimised legal",
        "BM25, chunk=384, overlap=0.2, top_k=20, chain-of-thought, temp=1.0",
        MicroPilotConfig(
            chunk_size=384, overlap=0.2, top_k=20,
            prompt_strategy="chain-of-thought", temperature=1.0,
        ),
        "ga_optimised_legal",
    ),
]

for name, config_str, pc, run_id in runner_configs:
    print(f"\n{'=' * 50}")
    print(f"Config: {name}")
    run_config = RunConfig(
        pipeline_config=pc,
        run_id=run_id,
        qa_records=legal_records,
        corpus_documents=corpus_documents,
        cache_dir=CACHE_DIR,
        results_dir=paths.results / "phase14" / "legal_baselines_runs",
        model_id=wrapper._model_id,
        seed=42,
    )
    result = runner.run(run_config)
    agg = result.aggregate_metrics
    comp = compute_fitness_components(agg, weights)
    print(f"  fitness={comp.fitness:.4f} F1={agg.mean_f1:.4f}")

    systems.append({
        "name": name,
        "config": config_str,
        "metrics": {
            "mean_em": agg.mean_em,
            "mean_f1": agg.mean_f1,
            "mean_citation_precision": agg.mean_citation_precision,
            "mean_citation_recall": agg.mean_citation_recall,
            "mean_citation_f1": agg.mean_citation_f1,
            "mean_retrieval_recall_at_k": agg.mean_retrieval_recall_at_k,
            "mean_input_tokens": agg.mean_input_tokens,
            "mean_latency_ms": agg.mean_latency_ms,
        },
        "fitness": comp.fitness,
    })

# ── Print comparison table ────────────────────────────────────────────────────
print(f"\n{'=' * 95}")
header = (
    f"{'System':<24} | {'EM':>6} | {'F1':>6} | {'CitP':>6} | "
    f"{'CitR':>6} | {'CitF1':>6} | {'RetR@k':>7} | {'Fitness':>8}"
)
print(header)
print("-" * 95)
for s in systems:
    m = s["metrics"]
    print(
        f"{s['name']:<24} | {m['mean_em']:>6.4f} | {m['mean_f1']:>6.4f} | "
        f"{m['mean_citation_precision']:>6.4f} | "
        f"{m['mean_citation_recall']:>6.4f} | "
        f"{m['mean_citation_f1']:>6.4f} | "
        f"{m['mean_retrieval_recall_at_k']:>7.4f} | "
        f"{s['fitness']:>8.4f}"
    )

summary = {
    "track": "legal",
    "n_records": len(legal_records),
    "fitness_weights": "legal (w_g=0.40)",
    "systems": systems,
}
OUT_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"\nSaved to {OUT_PATH}")

# Force exit — httpx connection pool in the openai/groq client holds
# non-daemon threads that prevent natural process termination.
sys.exit(0)
