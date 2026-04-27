"""Backfill Iterative-RAG pipeline_results.jsonl for the Phase 16 held-out set.

Re-runs Iterative-RAG across the same 300-record QA and Legal slices used
by run_held_out_phase16.py (the 2026-04-21 held-out pass) and writes a
pipeline_results.jsonl under each track's runs directory. This unlocks
RAGAS Faithfulness scoring for Iterative-RAG, which was previously
UNAVAILABLE because run_iterative_rag did not persist retrieved chunks.

Does NOT touch results/phase16/held_out_phase16.json or any other system's
outputs. The wrapper-level response cache at
data/cache/held_out_phase16_{track}/ should hit for every LLM call,
making this a fast re-materialisation rather than a full regeneration.

Run from project root (SambaNova recommended — matches the original pass):
    INFERENCE_PROVIDER=sambanova python scripts/backfill_iterative_rag_jsonl.py

Output:
    results/phase16/held_out_phase16_general_runs/
        iterative_rag_held_out_general/pipeline_results.jsonl
    results/phase16/held_out_phase16_legal_runs/
        iterative_rag_held_out_legal/pipeline_results.jsonl
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

os.environ["SNAP_CACHE_KEYS"] = "1"
os.environ.setdefault("ASYNC_EVAL", "0")

from rag_evo.data.hotpotqa_loader import load_records
from rag_evo.data.legal_loader import load_legal_records
from rag_evo.evaluation.harness import EvaluationHarness
from rag_evo.evaluation.iterative_rag import run_iterative_rag
from rag_evo.ga.fitness import FitnessWeights, legal_fitness_weights
from rag_evo.generation.wrapper import GeneratorWrapper
from rag_evo.utils.paths import get_project_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

GENERAL_SLICE = slice(100, 400)
LEGAL_SLICE = slice(50, 350)
N_RECORDS_PER_TRACK = 300


def main() -> int:
    paths = get_project_paths()
    general_cache = paths.root / "data" / "cache" / "held_out_phase16_general"
    legal_cache = paths.root / "data" / "cache" / "held_out_phase16_legal"
    general_results = (
        paths.results / "phase16" / "held_out_phase16_general_runs"
    )
    legal_results = paths.results / "phase16" / "held_out_phase16_legal_runs"

    logger.info("Loading expanded debug subsets...")
    general_records, general_corpus = load_records(
        Path("data/processed/hotpotqa_debug")
    )
    legal_records, legal_corpus = load_legal_records(
        Path("data/processed/legal_debug")
    )
    general_held_out = general_records[GENERAL_SLICE]
    legal_held_out = legal_records[LEGAL_SLICE]
    assert len(general_held_out) == N_RECORDS_PER_TRACK
    assert len(legal_held_out) == N_RECORDS_PER_TRACK
    logger.info(
        "General-QA: %d held-out  |  Legal: %d held-out",
        len(general_held_out), len(legal_held_out),
    )

    wrapper = GeneratorWrapper()

    tracks = [
        {
            "name": "general",
            "records": general_held_out,
            "corpus": general_corpus,
            "weights": FitnessWeights(),
            "cache_dir": general_cache / "iterative",
            "results_dir": general_results,
            "run_id": "iterative_rag_held_out_general",
        },
        {
            "name": "legal",
            "records": legal_held_out,
            "corpus": legal_corpus,
            "weights": legal_fitness_weights(),
            "cache_dir": legal_cache / "iterative",
            "results_dir": legal_results,
            "run_id": "iterative_rag_held_out_legal",
        },
    ]

    for track in tracks:
        logger.info(
            "=== Iterative-RAG backfill: %s  (%d records) ===",
            track["name"], len(track["records"]),
        )
        harness = EvaluationHarness(track["records"])
        summary = run_iterative_rag(
            qa_records=track["records"],
            corpus_documents=track["corpus"],
            config={},
            wrapper=wrapper,
            harness=harness,
            cache_dir=track["cache_dir"],
            fitness_weights=track["weights"],
            results_dir=track["results_dir"],
            run_id=track["run_id"],
        )
        jsonl = track["results_dir"] / track["run_id"] / "pipeline_results.jsonl"
        logger.info(
            "%s done: fitness=%.4f  round2=%d/%d  jsonl=%s",
            track["name"],
            summary["fitness"],
            summary["n_round2_triggered"],
            summary["n_records"],
            jsonl,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
