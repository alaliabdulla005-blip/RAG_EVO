"""Population size ablation — compare pop_size ∈ {5, 10, 20}.

Mini-GA: generations=5, mutation_rate=0.2, records=qa_records[:10], seed=42.
Each pop_size uses a separate cache and output directory.

Diversity metric: fraction of unique configs in the final generation
population (num_unique / pop_size). Computed from the saved
gen_004/population.json artifact.

Run from project root:
    python scripts/run_population_ablation.py

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

POP_SIZES = [5, 10, 20]
OUT_PATH = Path("results/ablations/population_size_ablation.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def _compute_diversity(pop_json_path: Path, pop_size: int) -> float:
    """Fraction of unique configs in the final generation population.

    Args:
        pop_json_path: Path to the population.json for the last generation.
        pop_size: Expected population size.

    Returns:
        Diversity score in [0.0, 1.0].
    """
    if not pop_json_path.exists():
        return 0.0
    population = json.loads(pop_json_path.read_text())
    unique_configs = {
        json.dumps(entry["config"], sort_keys=True)
        for entry in population
    }
    return len(unique_configs) / pop_size


print("Loading HotpotQA debug subset...")
all_records, corpus_documents = load_records(
    Path("data/processed/hotpotqa_debug")
)
records = all_records[:10]
print(f"Using {len(records)} records, {len(corpus_documents)} corpus docs")

wrapper = GeneratorWrapper()
ablation_results = []

for pop_size in POP_SIZES:
    print(f"\n{'=' * 50}")
    print(f"Population size = {pop_size}")

    cache = ResultCache(
        paths.root / "data" / "cache" / f"population_ablation_{pop_size}"
    )
    harness = EvaluationHarness(records)
    runner = ExperimentRunner(
        wrapper=wrapper, harness=harness, cache=cache
    )

    output_dir = paths.results / "ablations" / f"population_size_{pop_size}"

    ga_config = GAConfig(
        population_size=pop_size,
        generations=5,
        tournament_size=3,
        crossover_rate=0.5,
        mutation_rate=0.2,
        elite_count=1,
        seed=42,
    )

    template = RunConfig(
        pipeline_config=MicroPilotConfig(),
        run_id="template",
        qa_records=records,
        corpus_documents=corpus_documents,
        cache_dir=paths.root / "data" / "cache" / f"population_ablation_{pop_size}",
        results_dir=(
            paths.results / "ablations"
            / f"population_size_{pop_size}" / "runs"
        ),
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

    # Diversity from saved last-generation population artifact
    last_pop_path = output_dir / "gen_004" / "population.json"
    final_diversity = _compute_diversity(last_pop_path, pop_size)

    ablation_results.append({
        "population_size": pop_size,
        "best_fitness": best_fitness,
        "final_mean_fitness": final_mean_fitness,
        "final_diversity": final_diversity,
        "best_config": best_config,
    })

    print(
        f"  best_fitness={best_fitness:.4f} "
        f"final_mean={final_mean_fitness:.4f} "
        f"diversity={final_diversity:.3f}"
    )
    c = best_config
    print(f"  best_config: chunk={c['chunk_size']} overlap={c['overlap']} "
          f"top_k={c['top_k']} strategy={c['prompt_strategy']} "
          f"temp={c['temperature']:.3f}")

print(f"\n{'=' * 65}")
print(f"{'Pop':>5} | {'Best fit':>9} | {'Final mean':>11} | "
      f"{'Diversity':>9} | Best config summary")
print("-" * 75)
for r in ablation_results:
    c = r["best_config"]
    print(
        f"{r['population_size']:>5} | {r['best_fitness']:>9.4f} | "
        f"{r['final_mean_fitness']:>11.4f} | {r['final_diversity']:>9.3f} | "
        f"cs={c['chunk_size']} ov={c['overlap']} k={c['top_k']} "
        f"{c['prompt_strategy'][:4]} t={c['temperature']:.2f}"
    )

summary = {
    "ablation": "population_size",
    "values_tested": POP_SIZES,
    "results": ablation_results,
}
OUT_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"\nSaved to {OUT_PATH}")

# Force exit — httpx connection pool in the openai/groq client holds
# non-daemon threads that prevent natural process termination.
sys.exit(0)
