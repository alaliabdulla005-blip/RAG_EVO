"""Mutation rate ablation — compare mutation_rate ∈ {0.1, 0.2, 0.4}.

Mini-GA: population=5, generations=5, records=qa_records[:10], seed=42.
Each rate uses a separate cache and output directory.

Run from project root:
    python scripts/run_mutation_ablation.py

Requires INFERENCE_PROVIDER and matching API key in .env.
"""

import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

from rag_evo.cache.cache import ResultCache
from rag_evo.config.schema import MicroPilotConfig
from rag_evo.config.space import SearchSpace
from rag_evo.data.hotpotqa_loader import load_records
from rag_evo.evaluation.harness import EvaluationHarness
from rag_evo.ga.fitness import FitnessWeights
from rag_evo.ga.loop import GAConfig, GeneticAlgorithm
from rag_evo.generation.wrapper import GeneratorWrapper
from rag_evo.runner.experiment_runner import ExperimentRunner, RunConfig
from rag_evo.utils.paths import get_project_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

paths = get_project_paths()

MUTATION_RATES = [0.1, 0.2, 0.4]
OUT_PATH = Path("results/ablations/mutation_rate_ablation.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Loading HotpotQA debug subset...")
all_records, corpus_documents = load_records(
    Path("data/processed/hotpotqa_debug")
)
records = all_records[:10]
print(f"Using {len(records)} records, {len(corpus_documents)} corpus docs")

wrapper = GeneratorWrapper()
ablation_results = []

for rate in MUTATION_RATES:
    rate_str = str(rate).replace(".", "")
    print(f"\n{'=' * 50}")
    print(f"Mutation rate = {rate}")

    cache = ResultCache(
        paths.root / "data" / "cache" / f"mutation_ablation_{rate_str}"
    )
    harness = EvaluationHarness(records)
    runner = ExperimentRunner(
        wrapper=wrapper, harness=harness, cache=cache
    )

    output_dir = paths.results / "ablations" / f"mutation_rate_{rate_str}"

    ga_config = GAConfig(
        population_size=5,
        generations=5,
        tournament_size=3,
        crossover_rate=0.5,
        mutation_rate=rate,
        elite_count=1,
        seed=42,
    )

    template = RunConfig(
        pipeline_config=MicroPilotConfig(),
        run_id="template",
        qa_records=records,
        corpus_documents=corpus_documents,
        cache_dir=paths.root / "data" / "cache" / f"mutation_ablation_{rate_str}",
        results_dir=paths.results / "ablations" / f"mutation_rate_{rate_str}" / "runs",
        model_id=wrapper._model_id,
        seed=42,
    )

    ga = GeneticAlgorithm(
        ga_config=ga_config,
        run_config_template=template,
        runner=runner,
        search_space=SearchSpace("micro_pilot"),
        fitness_weights=FitnessWeights(),
        output_dir=output_dir,
    )

    gen_records = ga.run()

    best_fitness = max(r.best_fitness for r in gen_records)
    final_mean_fitness = gen_records[-1].mean_fitness
    best_record = max(gen_records, key=lambda r: r.best_fitness)
    best_config = best_record.best_config

    ablation_results.append({
        "mutation_rate": rate,
        "best_fitness": best_fitness,
        "final_mean_fitness": final_mean_fitness,
        "best_config": best_config,
    })

    print(
        f"  best_fitness={best_fitness:.4f} "
        f"final_mean={final_mean_fitness:.4f}"
    )
    print(f"  best_config: chunk={best_config['chunk_size']} "
          f"overlap={best_config['overlap']} top_k={best_config['top_k']} "
          f"strategy={best_config['prompt_strategy']} "
          f"temp={best_config['temperature']:.3f}")

print(f"\n{'=' * 60}")
print(f"{'Rate':>6} | {'Best fit':>10} | {'Final mean':>11} | Best config summary")
print("-" * 70)
for r in ablation_results:
    c = r["best_config"]
    print(
        f"{r['mutation_rate']:>6.1f} | {r['best_fitness']:>10.4f} | "
        f"{r['final_mean_fitness']:>11.4f} | "
        f"cs={c['chunk_size']} ov={c['overlap']} k={c['top_k']} "
        f"{c['prompt_strategy'][:4]} t={c['temperature']:.2f}"
    )

summary = {
    "ablation": "mutation_rate",
    "values_tested": MUTATION_RATES,
    "results": ablation_results,
}
OUT_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"\nSaved to {OUT_PATH}")

# Force exit — httpx connection pool in the openai/groq client holds
# non-daemon threads that prevent natural process termination.
sys.exit(0)
