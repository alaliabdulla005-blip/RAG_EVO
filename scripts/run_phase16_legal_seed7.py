"""Phase 16 — Legal track GA experiment, seed 7.

8-gene FullV2Config chromosome, pop=50, gen=100, DeepInfra provider.

Checkpoint/resume is automatic: if results/phase16/legal/seed_7/gen_NNN/
directories exist the GA resumes from the latest saved state.

DO NOT run this script automatically — run after provider calibration.

Run from project root:
    INFERENCE_PROVIDER=deepinfra python scripts/run_phase16_legal_seed7.py
"""

import json
import logging
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

# ── Verify all 48 dense indexes exist ──────────────────────────────────────
_INDEX_ROOT = Path("data/indexes")
_CHUNK_SIZES = [256, 384, 448, 512]
_OVERLAPS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
_TRACKS = ["general_qa", "legal"]
_missing = []
for _cs in _CHUNK_SIZES:
    for _ov in _OVERLAPS:
        for _tr in _TRACKS:
            _d = _INDEX_ROOT / f"dense_{_cs}_{_ov}_{_tr}"
            if not _d.exists() or not any(_d.iterdir()):
                _missing.append(str(_d))
if _missing:
    print("ERROR: Missing dense indexes detected.")
    for _m in _missing[:10]:
        print(f"  {_m}")
    if len(_missing) > 10:
        print(f"  ... and {len(_missing) - 10} more")
    print("\nRun: python scripts/build_dense_indexes.py")
    print("Then restart this script.")
    sys.exit(1)
print("Dense index check: all 48 indexes present.")

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
    max_workers=1,
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
