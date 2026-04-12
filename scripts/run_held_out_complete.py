"""Complete held-out evaluation — all systems on held-out test sets.

Evaluates every system in the dissertation on the held-out test subset
(records 50-149, n=100) that was never accessed during GA search.
This script reproduces all held-out results reported in Chapter 4.

Systems evaluated per track:
  1. Direct LLM (no retrieval)
  2. Naive RAG
  3. Conservative manual RAG
  4. Iterative RAG (2-round adaptive)
  5. GA-optimised (own-domain best)
  6. Cross-track transfer

Run from project root:
    INFERENCE_PROVIDER=sambanova python scripts/run_held_out_complete.py

Expected runtime: ~90-120 minutes (all systems, both tracks, 100 records each)
All results saved to results/phase15/held_out_complete.json
"""

import json
import logging
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

from rag_evo.cache.cache import ResultCache
from rag_evo.config.schema import MicroPilotConfig
from rag_evo.data.hotpotqa_loader import load_records
from rag_evo.data.legal_loader import load_legal_records
from rag_evo.evaluation.harness import EvaluationHarness
from rag_evo.evaluation.iterative_rag import run_iterative_rag
from rag_evo.ga.fitness import (
    FitnessWeights,
    compute_fitness_components,
    legal_fitness_weights,
)
from rag_evo.generation.config import DecodingConfig
from rag_evo.generation.wrapper import GeneratorWrapper
from rag_evo.pipeline.prompt_builder import SYSTEM_PROMPT_ZERO_SHOT
from rag_evo.pipeline.rag_pipeline import PipelineResult
from rag_evo.runner.experiment_runner import ExperimentRunner, RunConfig
from rag_evo.utils.paths import get_project_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

_CITATION_RE = re.compile(r"\[DOC_\d+\]")

paths = get_project_paths()

OUT_PATH = Path("results/phase15/held_out_complete.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Load 150-record subsets and slice held-out ────────────────────────────────
print("Loading 150-record subsets...")
general_records, general_corpus = load_records(Path("data/processed/hotpotqa_debug"))
legal_records, legal_corpus = load_legal_records(Path("data/processed/legal_debug"))

assert len(general_records) == 150, (
    f"Expected 150 General-QA records, got {len(general_records)}. "
    "Rebuild with scripts/rebuild_debug_subsets.py first."
)
assert len(legal_records) == 150, (
    f"Expected 150 legal records, got {len(legal_records)}. "
    "Rebuild with scripts/rebuild_debug_subsets.py first."
)

general_held_out = general_records[50:]  # records 50-149
legal_held_out = legal_records[50:]      # records 50-149
assert len(general_held_out) == 100
assert len(legal_held_out) == 100

print(f"General-QA: {len(general_records)} total, {len(general_held_out)} held-out")
print(f"Legal: {len(legal_records)} total, {len(legal_held_out)} held-out")

wrapper = GeneratorWrapper()


# ── Helper: Direct LLM (no retrieval) ────────────────────────────────────────
def _eval_direct_llm(
    name: str,
    records: list,
    wrapper: GeneratorWrapper,
    harness: EvaluationHarness,
    weights: FitnessWeights,
) -> dict:
    """Evaluate the no-retrieval baseline by calling the wrapper directly."""
    direct_decoding = DecodingConfig(temperature=0.0, max_tokens=150, seed=42)
    direct_results: list[PipelineResult] = []

    for record in records:
        output = wrapper.generate(
            system_prompt=SYSTEM_PROMPT_ZERO_SHOT,
            context_block="[No context provided]",
            query=record.question,
            decoding=direct_decoding,
        )
        cited = _CITATION_RE.findall(output.answer)
        direct_results.append(
            PipelineResult(
                record_id=record.record_id,
                question=record.question,
                answer=output.answer,
                cited_chunk_ids=cited,
                retrieved_chunks=[],
                input_tokens=output.input_tokens,
                output_tokens=output.output_tokens,
                latency_ms=output.latency_ms,
                retriever_type="none",
                prompt_strategy="zero-shot",
                chunk_size=0,
                overlap=0.0,
                top_k=0,
                raw_prompt=output.raw_prompt,
                raw_answer=output.raw_answer,
            )
        )

    _, agg = harness.evaluate_batch(direct_results)
    comp = compute_fitness_components(agg, weights)
    print(f"  fitness={comp.fitness:.4f} F1={agg.mean_f1:.4f}")
    return {
        "name": name,
        "config": "no retrieval, zero-shot, temperature=0.0",
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
        "fitness": comp.fitness,
    }


# ── Helper: Standard RAG via ExperimentRunner ────────────────────────────────
def _eval_runner_config(
    name: str,
    pc: MicroPilotConfig,
    run_id: str,
    records: list,
    corpus: list,
    wrapper: GeneratorWrapper,
    weights: FitnessWeights,
    cache_dir: Path,
    results_dir: Path,
) -> dict:
    """Evaluate a MicroPilotConfig on given records using ExperimentRunner."""
    harness = EvaluationHarness(records)
    cache = ResultCache(cache_dir)
    runner = ExperimentRunner(wrapper=wrapper, harness=harness, cache=cache)
    run_config = RunConfig(
        pipeline_config=pc,
        run_id=run_id,
        qa_records=records,
        corpus_documents=corpus,
        cache_dir=cache_dir,
        results_dir=results_dir,
        model_id=wrapper._model_id,
        seed=42,
    )
    result = runner.run(run_config)
    agg = result.aggregate_metrics
    comp = compute_fitness_components(agg, weights)
    print(f"  fitness={comp.fitness:.4f} F1={agg.mean_f1:.4f}")
    return {
        "name": name,
        "config": {
            "chunk_size": pc.chunk_size,
            "overlap": pc.overlap,
            "top_k": pc.top_k,
            "prompt_strategy": pc.prompt_strategy,
            "temperature": pc.temperature,
        },
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
        "fitness": comp.fitness,
    }


# ── Helper: Iterative RAG ─────────────────────────────────────────────────────
def _eval_iterative_rag(
    name: str,
    records: list,
    corpus: list,
    wrapper: GeneratorWrapper,
    weights: FitnessWeights,
    cache_dir: Path,
) -> dict:
    """Evaluate 2-round adaptive iterative RAG on held-out records."""
    harness = EvaluationHarness(records)
    result = run_iterative_rag(
        qa_records=records,
        corpus_documents=corpus,
        config={},  # defaults: chunk=384, overlap=0.3, top_k=10, temp=0.0
        wrapper=wrapper,
        harness=harness,
        cache_dir=cache_dir,
        fitness_weights=weights,
    )
    m = result["metrics"]
    print(
        f"  fitness={result['fitness']:.4f} F1={m['mean_f1']:.4f}"
        f"  Round2={result['n_round2_triggered']}/{len(records)}"
    )
    return {
        "name": name,
        "config": result["config"],
        "metrics": m,
        "fitness": result["fitness"],
    }


# ── General-QA held-out ───────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("General-QA held-out evaluation (records 50-149, n=100)")

general_weights = FitnessWeights()
general_cache = paths.root / "data" / "cache" / "held_out_complete_general"
general_results_dir = paths.results / "phase15" / "held_out_complete_general_runs"
general_held_harness = EvaluationHarness(general_held_out)

general_systems: list[dict] = []

print("\n[1/6] Direct LLM (General-QA held-out, n=100)...")
general_systems.append(
    _eval_direct_llm(
        name="Direct LLM",
        records=general_held_out,
        wrapper=wrapper,
        harness=general_held_harness,
        weights=general_weights,
    )
)

print("\n[2/6] Naive RAG (General-QA held-out, n=100)...")
general_systems.append(
    _eval_runner_config(
        name="Naive RAG",
        pc=MicroPilotConfig(
            chunk_size=256, overlap=0.0, top_k=5,
            prompt_strategy="zero-shot", temperature=0.0,
        ),
        run_id="naive_rag_held_out_complete_general",
        records=general_held_out,
        corpus=general_corpus,
        wrapper=wrapper,
        weights=general_weights,
        cache_dir=general_cache,
        results_dir=general_results_dir,
    )
)

print("\n[3/6] Conservative manual RAG (General-QA held-out, n=100)...")
general_systems.append(
    _eval_runner_config(
        name="Conservative manual RAG",
        pc=MicroPilotConfig(
            chunk_size=256, overlap=0.0, top_k=3,
            prompt_strategy="zero-shot", temperature=0.0,
        ),
        run_id="conservative_rag_held_out_complete_general",
        records=general_held_out,
        corpus=general_corpus,
        wrapper=wrapper,
        weights=general_weights,
        cache_dir=general_cache,
        results_dir=general_results_dir,
    )
)

print("\n[4/6] Iterative RAG (General-QA held-out, n=100)...")
general_systems.append(
    _eval_iterative_rag(
        name="Iterative RAG",
        records=general_held_out,
        corpus=general_corpus,
        wrapper=wrapper,
        weights=general_weights,
        cache_dir=general_cache / "iterative",
    )
)

print("\n[5/6] GA-optimised Phase 12 (General-QA held-out, n=100)...")
general_systems.append(
    _eval_runner_config(
        name="GA-optimised (Phase 12)",
        pc=MicroPilotConfig(
            chunk_size=512, overlap=0.0, top_k=19,
            prompt_strategy="chain-of-thought",
            temperature=0.6743266807244299,
        ),
        run_id="ga_phase12_held_out_complete_general",
        records=general_held_out,
        corpus=general_corpus,
        wrapper=wrapper,
        weights=general_weights,
        cache_dir=general_cache,
        results_dir=general_results_dir,
    )
)

print("\n[6/6] Phase 13 Legal→General transfer (General-QA held-out, n=100)...")
general_systems.append(
    _eval_runner_config(
        name="Phase 13 Legal→General transfer",
        pc=MicroPilotConfig(
            chunk_size=384, overlap=0.2, top_k=20,
            prompt_strategy="chain-of-thought", temperature=1.0,
        ),
        run_id="legal_transfer_held_out_complete_general",
        records=general_held_out,
        corpus=general_corpus,
        wrapper=wrapper,
        weights=general_weights,
        cache_dir=general_cache,
        results_dir=general_results_dir,
    )
)

# ── Legal held-out ────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("Legal held-out evaluation (records 50-149, n=100)")

legal_weights = legal_fitness_weights()
legal_cache = paths.root / "data" / "cache" / "held_out_complete_legal"
legal_results_dir = paths.results / "phase15" / "held_out_complete_legal_runs"
legal_held_harness = EvaluationHarness(legal_held_out)

legal_systems: list[dict] = []

print("\n[1/6] Direct LLM (Legal held-out, n=100)...")
legal_systems.append(
    _eval_direct_llm(
        name="Direct LLM",
        records=legal_held_out,
        wrapper=wrapper,
        harness=legal_held_harness,
        weights=legal_weights,
    )
)

print("\n[2/6] Naive RAG (Legal held-out, n=100)...")
legal_systems.append(
    _eval_runner_config(
        name="Naive RAG",
        pc=MicroPilotConfig(
            chunk_size=256, overlap=0.0, top_k=5,
            prompt_strategy="zero-shot", temperature=0.0,
        ),
        run_id="naive_rag_held_out_complete_legal",
        records=legal_held_out,
        corpus=legal_corpus,
        wrapper=wrapper,
        weights=legal_weights,
        cache_dir=legal_cache,
        results_dir=legal_results_dir,
    )
)

print("\n[3/6] Conservative manual RAG (Legal held-out, n=100)...")
legal_systems.append(
    _eval_runner_config(
        name="Conservative manual RAG",
        pc=MicroPilotConfig(
            chunk_size=256, overlap=0.0, top_k=3,
            prompt_strategy="zero-shot", temperature=0.0,
        ),
        run_id="conservative_rag_held_out_complete_legal",
        records=legal_held_out,
        corpus=legal_corpus,
        wrapper=wrapper,
        weights=legal_weights,
        cache_dir=legal_cache,
        results_dir=legal_results_dir,
    )
)

print("\n[4/6] Iterative RAG (Legal held-out, n=100)...")
legal_systems.append(
    _eval_iterative_rag(
        name="Iterative RAG",
        records=legal_held_out,
        corpus=legal_corpus,
        wrapper=wrapper,
        weights=legal_weights,
        cache_dir=legal_cache / "iterative",
    )
)

print("\n[5/6] GA-optimised Phase 13 Legal (Legal held-out, n=100)...")
legal_systems.append(
    _eval_runner_config(
        name="GA-optimised (Phase 13 Legal)",
        pc=MicroPilotConfig(
            chunk_size=384, overlap=0.2, top_k=20,
            prompt_strategy="chain-of-thought", temperature=1.0,
        ),
        run_id="ga_phase13_held_out_complete_legal",
        records=legal_held_out,
        corpus=legal_corpus,
        wrapper=wrapper,
        weights=legal_weights,
        cache_dir=legal_cache,
        results_dir=legal_results_dir,
    )
)

print("\n[6/6] Phase 12 General→Legal transfer (Legal held-out, n=100)...")
legal_systems.append(
    _eval_runner_config(
        name="Phase 12 General→Legal transfer",
        pc=MicroPilotConfig(
            chunk_size=512, overlap=0.0, top_k=19,
            prompt_strategy="chain-of-thought",
            temperature=0.6743266807244299,
        ),
        run_id="general_transfer_held_out_complete_legal",
        records=legal_held_out,
        corpus=legal_corpus,
        wrapper=wrapper,
        weights=legal_weights,
        cache_dir=legal_cache,
        results_dir=legal_results_dir,
    )
)

# ── Print summary table ───────────────────────────────────────────────────────
print(f"\n{'=' * 72}")
print(
    f"{'System':<32} | {'QA held-out':>11} | {'Legal held-out':>14}"
)
print("-" * 72)
for qa_row, leg_row in zip(general_systems, legal_systems):
    print(
        f"{qa_row['name']:<32} | {qa_row['fitness']:>11.4f}"
        f" | {leg_row['fitness']:>14.4f}"
    )

# ── Save JSON ─────────────────────────────────────────────────────────────────
output = {
    "evaluation_protocol": "held-out test set, records 50-149, n=100",
    "seed": 42,
    "general_qa": {
        "weights": "general (w_q=0.40)",
        "n_records": 100,
        "record_range": "50-149",
        "systems": general_systems,
    },
    "legal": {
        "weights": "legal (w_g=0.40)",
        "n_records": 100,
        "record_range": "50-149",
        "systems": legal_systems,
    },
}
OUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
print(f"\nSaved to {OUT_PATH}")

# Force exit — httpx connection pool in the openai/groq client holds
# non-daemon threads that prevent natural process termination.
sys.exit(0)
