"""Phase 13 — Legal track GA experiments (CUAD).

Population=20, Generations=15, 50 legal records, SambaNova.

Run from project root:
    INFERENCE_PROVIDER=sambanova python scripts/run_phase13_legal_qa.py

Uses parallel evaluation (4 workers) — estimated runtime: 3-5 hours.
Checkpoints saved after every generation — safe to Ctrl+C and resume.
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
from rag_evo.data.legal_loader import load_legal_records
from rag_evo.evaluation.harness import EvaluationHarness
from rag_evo.ga.fitness import legal_fitness_weights
from rag_evo.ga.loop import GAConfig, GeneticAlgorithm
from rag_evo.generation.wrapper import GeneratorWrapper
from rag_evo.runner.experiment_runner import ExperimentRunner, RunConfig
from rag_evo.utils.paths import get_project_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

paths = get_project_paths()

# ── Startup banner ────────────────────────────────────────────────────────────
print("=" * 65)
print("PHASE 13 — LEGAL TRACK GA EXPERIMENT")
print("=" * 65)
print("  Dataset          : CUAD (theatticusproject/cuad-qa)")
print("  Population size  : 20")
print("  Generations      : 15")
print("  Crossover rate   : 0.5")
print("  Mutation rate    : 0.2")
print("  Elite count      : 1")
print("  Parallel workers : 4")
print("  Legal fitness weights: w_g=0.40 (grounding priority)")
print("  Seed             : 42")
print("  Checkpoint dir   : results/phase13/legal_qa/")
print("=" * 65)

# ── Load data ────────────────────────────────────────────────────────────────
print("\nLoading legal debug subset ...")
legal_records, corpus_documents = load_legal_records(
    Path("data/processed/legal_debug")
)
print(
    f"Loaded {len(legal_records)} legal QA records, "
    f"{len(corpus_documents)} corpus docs"
)

# ── Build components ─────────────────────────────────────────────────────────
wrapper = GeneratorWrapper()
harness = EvaluationHarness(legal_records)
cache = ResultCache(
    paths.root / "data" / "cache" / "phase13_legal_qa"
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
    max_workers=4,
)

output_dir = paths.results / "phase13" / "legal_qa"
output_dir.mkdir(parents=True, exist_ok=True)

template = RunConfig(
    pipeline_config=MicroPilotConfig(),
    run_id="template",
    qa_records=legal_records,
    corpus_documents=corpus_documents,
    cache_dir=paths.root / "data" / "cache" / "phase13_legal_qa",
    results_dir=paths.results / "phase13" / "legal_qa" / "runs",
    model_id=wrapper._model_id,
    seed=42,
)

weights = legal_fitness_weights()

ga = GeneticAlgorithm(
    ga_config=ga_config,
    run_config_template=template,
    runner=runner,
    search_space=SearchSpace("micro_pilot"),
    fitness_weights=weights,
    output_dir=output_dir,
)

# ── Run (auto-resumes from checkpoint if gen_NNN/ dirs exist) ────────────────
print(
    "\nStarting Phase 13 GA run "
    "(pop=20, gen=15, crossover=0.5, workers=4) ..."
)
records = ga.run()

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 13 GA COMPLETE")
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
print(
    f"\nOverall best fitness: {best.best_fitness:.4f} "
    f"(gen {best.generation_idx})"
)
print(f"Best config: {json.dumps(best.best_config, indent=2)}")

summary_path = output_dir / "phase13_summary.json"
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
