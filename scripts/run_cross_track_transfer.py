"""Cross-track config transfer evaluation — Phase 14.

Evaluates two cross-track transfers:
  1. Phase 12 General-QA best config on legal data
     Config: chunk=512, overlap=0.0, top_k=19, CoT, temp=0.674
     Fitness weights: legal (w_g=0.40)
     Question: does the General-QA optimal config transfer to legal?

  2. Phase 13 Legal best config on General-QA data
     Config: chunk=384, overlap=0.2, top_k=20, CoT, temp=1.0
     Fitness weights: general (w_q=0.40)
     Question: does the Legal optimal config transfer to General-QA?

Each transfer uses 50 records from the respective track's debug subset.

Run from project root:
    INFERENCE_PROVIDER=sambanova python scripts/run_cross_track_transfer.py

Requires INFERENCE_PROVIDER and matching API key in .env.
DO NOT run this script — run it manually after legal baselines complete.
"""

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

from rag_evo.cache.cache import ResultCache
from rag_evo.config.schema import MicroPilotConfig
from rag_evo.data.hotpotqa_loader import load_records
from rag_evo.data.legal_loader import load_legal_records
from rag_evo.evaluation.harness import EvaluationHarness
from rag_evo.ga.fitness import (
    FitnessWeights,
    compute_fitness_components,
    legal_fitness_weights,
)
from rag_evo.generation.wrapper import GeneratorWrapper
from rag_evo.runner.experiment_runner import ExperimentRunner, RunConfig
from rag_evo.utils.paths import get_project_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

paths = get_project_paths()

CACHE_DIR = paths.root / "data" / "cache" / "cross_track_transfer"
OUT_PATH = Path("results/phase14/cross_track_transfer.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Load both datasets ────────────────────────────────────────────────────────
print("Loading legal debug subset...")
legal_records, legal_corpus = load_legal_records(
    Path("data/processed/legal_debug")
)
print(f"  {len(legal_records)} legal records, {len(legal_corpus)} corpus docs")

print("Loading General-QA debug subset...")
general_records, general_corpus = load_records(
    Path("data/processed/hotpotqa_debug")
)
print(f"  {len(general_records)} general records, {len(general_corpus)} corpus docs")

# ── Shared wrapper and cache ──────────────────────────────────────────────────
wrapper = GeneratorWrapper()
cache = ResultCache(CACHE_DIR)

# ── Transfer 1: General-QA best → legal data ─────────────────────────────────
print("\n" + "=" * 60)
print("Transfer 1: Phase 12 General-QA best config → Legal data")

general_best_config = MicroPilotConfig(
    chunk_size=512,
    overlap=0.0,
    top_k=19,
    prompt_strategy="chain-of-thought",
    temperature=0.6743266807244299,
)
legal_weights = legal_fitness_weights()
legal_harness = EvaluationHarness(legal_records)
legal_runner = ExperimentRunner(
    wrapper=wrapper, harness=legal_harness, cache=cache
)

t1_run = RunConfig(
    pipeline_config=general_best_config,
    run_id="general_on_legal",
    qa_records=legal_records,
    corpus_documents=legal_corpus,
    cache_dir=CACHE_DIR,
    results_dir=paths.results / "phase14" / "cross_track_runs",
    model_id=wrapper._model_id,
    seed=42,
)
t1_result = legal_runner.run(t1_run)
t1_agg = t1_result.aggregate_metrics
t1_comp = compute_fitness_components(t1_agg, legal_weights)
print(f"  fitness={t1_comp.fitness:.4f} F1={t1_agg.mean_f1:.4f} "
      f"CitF1={t1_agg.mean_citation_f1:.4f}")

# ── Transfer 2: Legal best → General-QA data ─────────────────────────────────
print("\n" + "=" * 60)
print("Transfer 2: Phase 13 Legal best config → General-QA data")

legal_best_config = MicroPilotConfig(
    chunk_size=384,
    overlap=0.2,
    top_k=20,
    prompt_strategy="chain-of-thought",
    temperature=1.0,
)
general_weights = FitnessWeights()
general_harness = EvaluationHarness(general_records)
general_runner = ExperimentRunner(
    wrapper=wrapper, harness=general_harness, cache=cache
)

t2_run = RunConfig(
    pipeline_config=legal_best_config,
    run_id="legal_on_general",
    qa_records=general_records,
    corpus_documents=general_corpus,
    cache_dir=CACHE_DIR,
    results_dir=paths.results / "phase14" / "cross_track_runs",
    model_id=wrapper._model_id,
    seed=42,
)
t2_result = general_runner.run(t2_run)
t2_agg = t2_result.aggregate_metrics
t2_comp = compute_fitness_components(t2_agg, general_weights)
print(f"  fitness={t2_comp.fitness:.4f} F1={t2_agg.mean_f1:.4f} "
      f"CitF1={t2_agg.mean_citation_f1:.4f}")


def _agg_to_metrics(agg) -> dict:  # type: ignore[type-arg]
    return {
        "mean_em": agg.mean_em,
        "mean_f1": agg.mean_f1,
        "mean_citation_precision": agg.mean_citation_precision,
        "mean_citation_recall": agg.mean_citation_recall,
        "mean_citation_f1": agg.mean_citation_f1,
        "mean_retrieval_recall_at_k": agg.mean_retrieval_recall_at_k,
        "mean_input_tokens": agg.mean_input_tokens,
        "mean_latency_ms": agg.mean_latency_ms,
    }


summary = {
    "transfers": [
        {
            "config_source": "Phase 12 General-QA GA best",
            "evaluated_on": "legal",
            "config": asdict(general_best_config),
            "metrics": _agg_to_metrics(t1_agg),
            "fitness": t1_comp.fitness,
            "fitness_weights": "legal (w_g=0.40)",
        },
        {
            "config_source": "Phase 13 Legal GA best",
            "evaluated_on": "general",
            "config": asdict(legal_best_config),
            "metrics": _agg_to_metrics(t2_agg),
            "fitness": t2_comp.fitness,
            "fitness_weights": "general (w_q=0.40)",
        },
    ]
}

OUT_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"\nSaved to {OUT_PATH}")

# Force exit — httpx connection pool in the openai/groq client holds
# non-daemon threads that prevent natural process termination.
sys.exit(0)
