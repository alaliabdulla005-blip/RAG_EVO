"""Seed stability check — re-evaluate Phase 10 best config across 3 seeds.

Re-runs the best config from Phase 10 with seeds [42, 123, 7] on all
50 debug records via ExperimentRunner and computes fitness components.

Run from project root:
    python scripts/run_seed_stability.py

Requires INFERENCE_PROVIDER and matching API key in .env.
"""

import json
import logging
import math
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

from rag_evo.cache.cache import ResultCache
from rag_evo.config.schema import MicroPilotConfig
from rag_evo.data.hotpotqa_loader import load_records
from rag_evo.evaluation.harness import EvaluationHarness
from rag_evo.ga.fitness import FitnessWeights, compute_fitness_components
from rag_evo.generation.wrapper import GeneratorWrapper
from rag_evo.runner.experiment_runner import (
    ExperimentRunner,
    RunConfig,
    generate_run_id,
)
from rag_evo.utils.paths import get_project_paths
from rag_evo.utils.seed import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

paths = get_project_paths()

BEST_CONFIG = MicroPilotConfig(
    chunk_size=384,
    overlap=0.3,
    top_k=20,
    prompt_strategy="chain-of-thought",
    temperature=0.7641852369412794,
)
SEEDS = [42, 123, 7]
OUT_PATH = Path("results/ablations/seed_stability.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Loading HotpotQA debug subset...")
debug_records, corpus_documents = load_records(
    Path("data/processed/hotpotqa_debug")
)
print(f"Loaded {len(debug_records)} records, {len(corpus_documents)} corpus docs")

wrapper = GeneratorWrapper()
weights = FitnessWeights()

seed_results = []

for seed in SEEDS:
    print(f"\n{'=' * 50}")
    print(f"Evaluating seed={seed} ...")
    set_global_seed(seed)

    cache = ResultCache(
        paths.root / "data" / "cache" / f"seed_stability_{seed}"
    )
    harness = EvaluationHarness(debug_records)
    runner = ExperimentRunner(
        wrapper=wrapper, harness=harness, cache=cache
    )

    run_id = generate_run_id(BEST_CONFIG)
    run_config = RunConfig(
        pipeline_config=BEST_CONFIG,
        run_id=run_id,
        qa_records=debug_records,
        corpus_documents=corpus_documents,
        cache_dir=paths.root / "data" / "cache" / f"seed_stability_{seed}",
        results_dir=paths.results / "ablations" / "seed_stability_runs",
        model_id=wrapper._model_id,
        seed=seed,
    )

    result = runner.run(run_config)
    components = compute_fitness_components(
        result.aggregate_metrics, weights
    )

    seed_results.append({
        "seed": seed,
        "fitness": components.fitness,
        "components": {
            "q_norm": components.q_norm,
            "g_norm": components.g_norm,
            "r_norm": components.r_norm,
            "c_norm": components.c_norm,
            "l_norm": components.l_norm,
        },
    })
    print(
        f"  seed={seed}: fitness={components.fitness:.4f} "
        f"Q={components.q_norm:.4f} G={components.g_norm:.4f} "
        f"R={components.r_norm:.4f} C={components.c_norm:.4f} "
        f"L={components.l_norm:.4f}"
    )

fitnesses = [r["fitness"] for r in seed_results]
mean_fitness = sum(fitnesses) / len(fitnesses)
variance = sum((f - mean_fitness) ** 2 for f in fitnesses) / len(fitnesses)
std_fitness = math.sqrt(variance)
cv_fitness = std_fitness / mean_fitness if mean_fitness != 0 else 0.0

print(f"\n{'=' * 50}")
print(f"{'Seed':>6} | {'Fitness':>8} | {'Q_norm':>7} | {'G_norm':>7} | "
      f"{'R_norm':>7} | {'C_norm':>7} | {'L_norm':>7}")
print("-" * 65)
for r in seed_results:
    c = r["components"]
    print(f"{r['seed']:>6} | {r['fitness']:>8.4f} | {c['q_norm']:>7.4f} | "
          f"{c['g_norm']:>7.4f} | {c['r_norm']:>7.4f} | "
          f"{c['c_norm']:>7.4f} | {c['l_norm']:>7.4f}")
print("-" * 65)
print(f"{'Mean':>6} | {mean_fitness:>8.4f}")
print(f"{'Std':>6} | {std_fitness:>8.4f}")
print(f"{'CV':>6} | {cv_fitness:>8.4f}")

summary = {
    "best_config": {
        "chunk_size": BEST_CONFIG.chunk_size,
        "overlap": BEST_CONFIG.overlap,
        "top_k": BEST_CONFIG.top_k,
        "prompt_strategy": BEST_CONFIG.prompt_strategy,
        "temperature": BEST_CONFIG.temperature,
    },
    "seeds": SEEDS,
    "results": seed_results,
    "mean_fitness": mean_fitness,
    "std_fitness": std_fitness,
    "cv_fitness": cv_fitness,
}
OUT_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"\nSaved to {OUT_PATH}")

# Force exit — httpx connection pool in the openai/groq client holds
# non-daemon threads that prevent natural process termination.
sys.exit(0)
