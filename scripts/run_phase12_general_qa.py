"""Phase 12 — Full GA experiment: pop=20, gen=15, all 50 debug records.

Key differences from the Phase 10 pilot:
  population_size = 20  (was 10)
  crossover_rate  = 0.5 (was 0.8)
  All 50 debug records used (pilot used all 50 too, but this is the
  formal Phase 12 run with fresh cache and output directory).

Checkpoint/resume is automatic: if results/phase12/general_qa/gen_NNN/
directories exist the GA resumes from the latest saved state.

Run from project root:
    python scripts/run_phase12_general_qa.py

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

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading HotpotQA debug subset...")
debug_records, corpus_documents = load_records(
    Path("data/processed/hotpotqa_debug")
)
print(
    f"Debug subset: {len(debug_records)} records, "
    f"{len(corpus_documents)} corpus docs"
)

# ── Build components ─────────────────────────────────────────────────────────
wrapper = GeneratorWrapper()
harness = EvaluationHarness(debug_records)
cache = ResultCache(
    paths.root / "data" / "cache" / "phase12_general_qa"
)
runner = ExperimentRunner(
    wrapper=wrapper, harness=harness, cache=cache
)

# ── GA setup ─────────────────────────────────────────────────────────────────
ga_config = GAConfig(
    population_size=20,
    generations=15,
    tournament_size=3,
    crossover_rate=0.5,
    mutation_rate=0.2,
    elite_count=1,
    seed=42,
)

output_dir = paths.results / "phase12" / "general_qa"
output_dir.mkdir(parents=True, exist_ok=True)

template = RunConfig(
    pipeline_config=MicroPilotConfig(),
    run_id="template",
    qa_records=debug_records,
    corpus_documents=corpus_documents,
    cache_dir=paths.root / "data" / "cache" / "phase12_general_qa",
    results_dir=paths.results / "phase12" / "general_qa" / "runs",
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

# ── Run (auto-resumes from checkpoint if gen_NNN/ dirs exist) ────────────────
print("Starting Phase 12 GA run (pop=20, gen=15, crossover=0.5) ...")
records = ga.run()

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 12 GA COMPLETE")
print("=" * 60)
for r in records:
    print(
        f"  Gen {r.generation_idx:2d}: "
        f"best={r.best_fitness:.4f} "
        f"mean={r.mean_fitness:.4f} "
        f"worst={r.worst_fitness:.4f} "
        f"({r.elapsed_s:.1f}s)"
    )

best = max(records, key=lambda r: r.best_fitness)
print(f"\nOverall best fitness: {best.best_fitness:.4f} "
      f"(gen {best.generation_idx})")
print(f"Best config: {json.dumps(best.best_config, indent=2)}")

summary_path = output_dir / "phase12_summary.json"
summary_path.write_text(
    json.dumps(
        [asdict(r) for r in records],
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(f"\nSummary saved to {summary_path}")

# Force exit — httpx connection pool in the openai/groq client holds
# non-daemon threads that prevent natural process termination.
sys.exit(0)
