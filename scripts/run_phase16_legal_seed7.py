"""Phase 16 — Legal track GA experiment, seed 7.

Phase 16 search space: 6 active genes (chunk_size, overlap, top_k,
prompt_strategy, temperature, max_output_tokens). retriever_type
fixed to BM25 and reranker_enabled fixed to False due to evaluation
time constraints on 8GB RAM with 100 dev records. Dense retrieval
and reranking are proposed as future work in Section 5.2.

8-gene FullV2Config chromosome (2 genes fixed, 6 searched),
pop=50, gen=100, DeepInfra provider.

Checkpoint/resume is automatic: if results/phase16/legal/seed_7/gen_NNN/
directories exist the GA resumes from the latest saved state.

DO NOT run this script automatically — run after provider calibration.

Run from project root:
    INFERENCE_PROVIDER=deepinfra python scripts/run_phase16_legal_seed7.py

IMPORTANT: Run with caffeinate to prevent macOS sleep:
  caffeinate -dims INFERENCE_PROVIDER=deepinfra \\
    python scripts/run_phase16_legal_seed7.py
Without caffeinate the laptop sleeps mid-run,
killing TCP connections and hanging the process.
"""

import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

from rag_evo.cache.cache import ResultCache
from rag_evo.config.schema import FullV2Config
from rag_evo.config.space import SearchSpace
from rag_evo.data.legal_loader import load_legal_records
from rag_evo.evaluation.harness import EvaluationHarness
from rag_evo.ga.fitness import legal_fitness_weights
from rag_evo.ga.loop import GAConfig, GeneticAlgorithm
from rag_evo.generation.wrapper import GeneratorWrapper
from rag_evo.retrieval.reranker import Reranker
from rag_evo.runner.experiment_runner import ExperimentRunner, RunConfig
from rag_evo.utils.paths import get_project_paths

SEED = 7

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

paths = get_project_paths()

# ── Verify BM25 corpus exists ──────────────────────────────────────────────
_CORPUS_DIR = Path("data/processed/legal_debug")
if not _CORPUS_DIR.exists() or not any(_CORPUS_DIR.iterdir()):
    print(f"ERROR: BM25 corpus not found at {_CORPUS_DIR}")
    print("Run the data processing pipeline before starting this experiment.")
    sys.exit(1)
print(f"BM25 corpus check: OK ({_CORPUS_DIR}).")

# ── Startup banner ────────────────────────────────────────────────────────────
print("=" * 65)
print(f"PHASE 16 — LEGAL TRACK GA EXPERIMENT (SEED {SEED})")
print("=" * 65)
print("  Chromosome        : FullV2Config (8 genes)")
print("  Provider          : DeepInfra (FP8)")
print("  Dataset           : CUAD (theatticusproject/cuad-qa)")
print("  Population size   : 50")
print("  Generations       : 100")
print("  Crossover rate    : 0.5")
print("  Mutation rate     : 0.2")
print("  Elite count       : 1")
print("  Dev set           : records[:50] (50 records)")
print("  Legal fitness     : w_g=0.40 (grounding priority)")
print(f"  Seed              : {SEED}")
print("  Cache (shared)    : data/cache/phase16_legal/")
print(f"  Output            : results/phase16/legal/seed_{SEED}/")
print("=" * 65)

# ── Load data ────────────────────────────────────────────────────────────────
print("\nLoading legal debug subset ...")
legal_records, corpus_documents = load_legal_records(
    Path("data/processed/legal_debug")
)
records = legal_records[:50]
print(
    f"Loaded {len(legal_records)} legal QA records, "
    f"{len(corpus_documents)} corpus docs — using first 50 records"
)

# ── Build components ─────────────────────────────────────────────────────────
wrapper = GeneratorWrapper()
harness = EvaluationHarness(records)
cache = ResultCache(
    paths.root / "data" / "cache" / "phase16_legal"
)
reranker = Reranker()
runner = ExperimentRunner(
    wrapper=wrapper, harness=harness, cache=cache, reranker=reranker
)

# ── GA setup ─────────────────────────────────────────────────────────────────
ga_config = GAConfig(
    population_size=50,
    generations=100,
    tournament_size=3,
    crossover_rate=0.5,
    mutation_rate=0.2,
    elite_count=1,
    seed=SEED,
    max_workers=int(os.environ.get("EVAL_WORKERS", "8")),
)

output_dir = paths.results / "phase16" / "legal" / f"seed_{SEED}"
output_dir.mkdir(parents=True, exist_ok=True)

template = RunConfig(
    pipeline_config=FullV2Config(),
    run_id="template",
    qa_records=records,
    corpus_documents=corpus_documents,
    cache_dir=paths.root / "data" / "cache" / "phase16_legal",
    results_dir=output_dir / "runs",
    model_id=wrapper._model_id,
    seed=SEED,
    track="legal",
    index_root=Path("data/indexes"),
)

weights = legal_fitness_weights()

ga = GeneticAlgorithm(
    ga_config=ga_config,
    run_config_template=template,
    runner=runner,
    search_space=SearchSpace("full_v2"),
    fitness_weights=weights,
    output_dir=output_dir,
)

# ── Run (auto-resumes from checkpoint if gen_NNN/ dirs exist) ────────────────
print(
    "\nStarting Phase 16 GA run "
    f"(pop=50, gen=100, crossover=0.5, seed={SEED}) ..."
)
ga_records = ga.run()

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"PHASE 16 LEGAL GA COMPLETE (SEED {SEED})")
print("=" * 60)
for r in ga_records:
    print(
        f"  Gen {r.generation_idx:2d}: "
        f"best={r.best_fitness:.4f} "
        f"mean={r.mean_fitness:.4f} "
        f"worst={r.worst_fitness:.4f} "
        f"({r.elapsed_s:.1f}s)"
    )

best = max(ga_records, key=lambda r: r.best_fitness)
print(
    f"\nOverall best fitness: {best.best_fitness:.4f} "
    f"(gen {best.generation_idx})"
)
print(f"Best config: {json.dumps(best.best_config, indent=2)}")

summary_dir = paths.results / "phase16" / "legal"
summary_dir.mkdir(parents=True, exist_ok=True)
summary_path = summary_dir / f"phase16_legal_seed{SEED}_summary.json"
summary_path.write_text(
    json.dumps(
        [asdict(r) for r in ga_records],
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(f"\nSummary saved to {summary_path}")

# Force exit — httpx connection pool in the openai/groq client holds
# non-daemon threads that prevent natural process termination.
sys.exit(0)
