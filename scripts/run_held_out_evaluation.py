"""Held-out evaluation — Phase 15, Task 2.

Evaluates best configs from Phase 12 (General-QA) and Phase 13 (Legal) on
records 50–149 (100 held-out per track) to confirm results generalise beyond
the 50-record development set used during GA search.

Split:
  Records 0–49:   Development set (GA searched on these)
  Records 50–149: Held-out test set (never seen during optimisation)

Evaluated configs per track:
  General-QA (records 50-149, general fitness weights):
    1. Naive RAG baseline
    2. Phase 12 GA best
    3. Phase 13 Legal GA best (cross-track transfer)

  Legal (records 50-149, legal fitness weights):
    1. Naive RAG baseline
    2. Phase 13 Legal GA best
    3. Phase 12 General-QA GA best (cross-track transfer)

Generalises? = abs(dev_fitness - held_out_fitness) / dev_fitness < 0.15

Run from project root:
    INFERENCE_PROVIDER=sambanova python scripts/run_held_out_evaluation.py

DO NOT run this script — run it manually after debug subsets are rebuilt to 150.
Requires INFERENCE_PROVIDER and matching API key in .env.
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

GENERALISATION_THRESHOLD = 0.15

OUT_PATH = Path("results/phase15/held_out_evaluation.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

LIMITATIONS = [
    (
        "Small development set (50 records) used for both GA search and final"
        " evaluation — no held-out test set during optimisation."
        " Evaluation noise CV~0.087."
    ),
    (
        "Temperature=1.0 for legal best config is potentially a noise artefact"
        " from stochastic evaluation on 50 records."
        " Temperature ablation (Task 1) investigates this."
    ),
    (
        "BM25-only retrieval throughout v1. Dense and hybrid retrieval are not"
        " explored. Retrieval recall on General-QA saturates at 0.730"
        " — a BM25 ceiling."
    ),
    (
        "Citation format mismatch on legal data: CUAD answers do not naturally"
        " elicit [DOC_N] citation markers, causing CitF1=0.123 despite"
        " RetR@5=0.940."
    ),
    (
        "Legal track GA interrupted by SambaNova 20M/day quota exhaustion."
        " Resumed on new account — results valid but infrastructure instability"
        " documented."
    ),
    (
        "5-gene search space only (chunk_size, overlap, top_k, prompt_strategy,"
        " temperature). retriever_type, reranker_enabled, max_output_tokens not"
        " searched in v1."
    ),
]

paths = get_project_paths()

# ── Load all 150 records for both tracks ─────────────────────────────────────
print("Loading 150-record subsets...")
general_records, general_corpus = load_records(
    Path("data/processed/hotpotqa_debug")
)
legal_records, legal_corpus = load_legal_records(
    Path("data/processed/legal_debug")
)

assert len(general_records) == 150, (
    f"Expected 150 General-QA records, got {len(general_records)}. "
    "Rebuild subsets with scripts/rebuild_debug_subsets.py first."
)
assert len(legal_records) == 150, (
    f"Expected 150 legal records, got {len(legal_records)}. "
    "Rebuild subsets with scripts/rebuild_debug_subsets.py first."
)

# Held-out: records 50–149
general_held_out = general_records[50:]
legal_held_out = legal_records[50:]
assert len(general_held_out) == 100
assert len(legal_held_out) == 100

print(f"General-QA: {len(general_records)} total, {len(general_held_out)} held-out")
print(f"Legal: {len(legal_records)} total, {len(legal_held_out)} held-out")

wrapper = GeneratorWrapper()


def _eval_config(
    name: str,
    pc: MicroPilotConfig,
    run_id: str,
    held_out_records: list,
    corpus: list,
    weights: FitnessWeights,
    cache_dir: Path,
    results_dir: Path,
    dev_fitness: float,
) -> dict:
    """Run one config on held-out records and compute generalisation gap."""
    harness = EvaluationHarness(held_out_records)
    cache = ResultCache(cache_dir)
    runner = ExperimentRunner(wrapper=wrapper, harness=harness, cache=cache)
    run_config = RunConfig(
        pipeline_config=pc,
        run_id=run_id,
        qa_records=held_out_records,
        corpus_documents=corpus,
        cache_dir=cache_dir,
        results_dir=results_dir,
        model_id=wrapper._model_id,
        seed=42,
    )
    result = runner.run(run_config)
    agg = result.aggregate_metrics
    comp = compute_fitness_components(agg, weights)

    held_fitness = comp.fitness
    if dev_fitness > 0:
        gap = abs(dev_fitness - held_fitness) / dev_fitness
    else:
        gap = float("inf")
    generalises = gap < GENERALISATION_THRESHOLD

    print(
        f"  {name}: dev={dev_fitness:.3f} held={held_fitness:.3f}"
        f" gap={gap:.3f} {'✓' if generalises else '✗'}"
    )

    return {
        "name": name,
        "config": asdict(pc),
        "dev_fitness": dev_fitness,
        "held_out_fitness": held_fitness,
        "generalisation_gap": gap,
        "generalises": generalises,
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
    }


# ── General-QA held-out ───────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("General-QA held-out evaluation (records 50–149, n=100)")
general_weights = FitnessWeights()
general_cache_base = paths.root / "data" / "cache" / "held_out_general"
general_results_dir = paths.results / "phase15" / "held_out_general_runs"

general_configs = [
    (
        "Naive RAG",
        MicroPilotConfig(
            chunk_size=256, overlap=0.0, top_k=5,
            prompt_strategy="zero-shot", temperature=0.0,
        ),
        "naive_rag_held_out_general",
        0.107,
    ),
    (
        "Phase 12 GA best",
        MicroPilotConfig(
            chunk_size=512, overlap=0.0, top_k=19,
            prompt_strategy="chain-of-thought",
            temperature=0.6743266807244299,
        ),
        "ga_phase12_held_out_general",
        0.240,
    ),
    (
        "Phase 13 Legal transfer",
        MicroPilotConfig(
            chunk_size=384, overlap=0.2, top_k=20,
            prompt_strategy="chain-of-thought", temperature=1.0,
        ),
        "legal_transfer_held_out_general",
        0.216,
    ),
]

general_held_results = []
for name, pc, run_id, dev_fitness in general_configs:
    row = _eval_config(
        name=name,
        pc=pc,
        run_id=run_id,
        held_out_records=general_held_out,
        corpus=general_corpus,
        weights=general_weights,
        cache_dir=general_cache_base,
        results_dir=general_results_dir,
        dev_fitness=dev_fitness,
    )
    general_held_results.append(row)

# ── Legal held-out ────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("Legal held-out evaluation (records 50–149, n=100)")
legal_weights = legal_fitness_weights()
legal_cache_base = paths.root / "data" / "cache" / "held_out_legal"
legal_results_dir = paths.results / "phase15" / "held_out_legal_runs"

legal_configs = [
    (
        "Naive RAG",
        MicroPilotConfig(
            chunk_size=256, overlap=0.0, top_k=5,
            prompt_strategy="zero-shot", temperature=0.0,
        ),
        "naive_rag_held_out_legal",
        0.026,
    ),
    (
        "Phase 13 Legal GA best",
        MicroPilotConfig(
            chunk_size=384, overlap=0.2, top_k=20,
            prompt_strategy="chain-of-thought", temperature=1.0,
        ),
        "ga_phase13_held_out_legal",
        0.185,
    ),
    (
        "Phase 12 General transfer",
        MicroPilotConfig(
            chunk_size=512, overlap=0.0, top_k=19,
            prompt_strategy="chain-of-thought",
            temperature=0.6743266807244299,
        ),
        "general_transfer_held_out_legal",
        0.186,
    ),
]

legal_held_results = []
for name, pc, run_id, dev_fitness in legal_configs:
    row = _eval_config(
        name=name,
        pc=pc,
        run_id=run_id,
        held_out_records=legal_held_out,
        corpus=legal_corpus,
        weights=legal_weights,
        cache_dir=legal_cache_base,
        results_dir=legal_results_dir,
        dev_fitness=dev_fitness,
    )
    legal_held_results.append(row)

# ── Print generalisation summary table ───────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"{'Config':<28} | {'Dev':>6} | {'Held-out':>8} | {'Δ':>6} | Generalises?")
print("-" * 80)
for r in general_held_results + legal_held_results:
    delta = r["held_out_fitness"] - r["dev_fitness"]
    gen_str = "Yes" if r["generalises"] else "No"
    print(
        f"{r['name']:<28} | {r['dev_fitness']:>6.3f} | "
        f"{r['held_out_fitness']:>8.3f} | {delta:>+6.3f} | {gen_str}"
    )

# ── Save JSON ─────────────────────────────────────────────────────────────────
output = {
    "generalisation_threshold": GENERALISATION_THRESHOLD,
    "n_held_out_per_track": 100,
    "record_range": "50-149",
    "limitations": LIMITATIONS,
    "general_qa": {
        "weights": "general (w_q=0.40)",
        "results": general_held_results,
    },
    "legal": {
        "weights": "legal (w_g=0.40)",
        "results": legal_held_results,
    },
}
OUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
print(f"\nSaved to {OUT_PATH}")

# Force exit — httpx connection pool in openai/groq client holds
# non-daemon threads that prevent natural process termination.
sys.exit(0)
