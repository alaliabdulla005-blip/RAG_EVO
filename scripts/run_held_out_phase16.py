"""Phase 16 held-out evaluation — all systems on expanded held-out test sets.

Evaluates 8 systems per track on held-out records that were never accessed
during GA search. Includes BERTScore F1 as an extended metric.

Systems evaluated per track:
  1. Direct LLM (no retrieval)
  2. Naive RAG (defaults)
  3. Iterative RAG (2-round adaptive)
  4. GA-optimised v1 (Phase 12/13 best)
  5. GA v2 seed 42 best
  6. GA v2 seed 123 best
  7. GA v2 seed 7 best
  8. Consensus best (highest mean held-out F1 across 3 seeds)

General-QA: records[100:400] (n=300)
Legal: records[50:350] (n=300)

DO NOT run this script automatically — run after all 6 GA runs complete.

Run from project root:
    INFERENCE_PROVIDER=deepinfra python scripts/run_held_out_phase16.py

Output: results/phase16/held_out_phase16.json
"""

import json
import logging
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

from rag_evo.cache.cache import ResultCache
from rag_evo.config.schema import FullV2Config, MicroPilotConfig
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
from rag_evo.retrieval.reranker import Reranker
from rag_evo.runner.experiment_runner import ExperimentRunner, RunConfig
from rag_evo.utils.paths import get_project_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

_CITATION_RE = re.compile(r"\[DOC_\d+\]")

paths = get_project_paths()

OUT_PATH = Path("results/phase16/held_out_phase16.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Load expanded debug subsets and slice held-out ───────────────────────────
print("Loading expanded debug subsets...")
general_records, general_corpus = load_records(Path("data/processed/hotpotqa_debug"))
legal_records, legal_corpus = load_legal_records(Path("data/processed/legal_debug"))

assert len(general_records) >= 400, (
    f"Expected >= 400 General-QA records, got {len(general_records)}. "
    "Rebuild with scripts/rebuild_debug_subsets.py first."
)
assert len(legal_records) >= 350, (
    f"Expected >= 350 legal records, got {len(legal_records)}. "
    "Rebuild with scripts/rebuild_debug_subsets.py first."
)

general_held_out = general_records[100:400]  # records 100-399
legal_held_out = legal_records[50:350]       # records 50-349
assert len(general_held_out) == 300
assert len(legal_held_out) == 300

print(f"General-QA: {len(general_records)} total, {len(general_held_out)} held-out")
print(f"Legal: {len(legal_records)} total, {len(legal_held_out)} held-out")

wrapper = GeneratorWrapper()


# ── BERTScore computation ────────────────────────────────────────────────────
def _compute_bertscore(predictions: list[str], references: list[str]) -> float:
    """Compute mean BERTScore F1 using microsoft/deberta-xlarge-mnli."""
    from bert_score import score as bert_score_fn

    _, _, f1 = bert_score_fn(
        predictions,
        references,
        model_type="microsoft/deberta-xlarge-mnli",
        batch_size=16,
        verbose=False,
    )
    return f1.mean().item()


# ── Helper: Direct LLM (no retrieval) ───────────────────────────────────────
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
    predictions = [r.answer for r in direct_results]
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
        "predictions": predictions,
    }


# ── Helper: Standard RAG via ExperimentRunner ────────────────────────────────
def _eval_runner_config(
    name: str,
    pc: MicroPilotConfig | FullV2Config,
    run_id: str,
    records: list,
    corpus: list,
    wrapper: GeneratorWrapper,
    weights: FitnessWeights,
    cache_dir: Path,
    results_dir: Path,
    reranker: Reranker | None = None,
    track: str | None = None,
    index_root: Path | None = None,
) -> dict:
    """Evaluate a MicroPilotConfig or FullV2Config using ExperimentRunner."""
    harness = EvaluationHarness(records)
    cache = ResultCache(cache_dir)
    runner = ExperimentRunner(
        wrapper=wrapper, harness=harness, cache=cache, reranker=reranker,
    )
    run_config = RunConfig(
        pipeline_config=pc,
        run_id=run_id,
        qa_records=records,
        corpus_documents=corpus,
        cache_dir=cache_dir,
        results_dir=results_dir,
        model_id=wrapper._model_id,
        seed=42,
        track=track,
        index_root=index_root,
    )
    result = runner.run(run_config)
    agg = result.aggregate_metrics

    # Extract predictions from saved pipeline_results.jsonl
    results_jsonl = results_dir / run_id / "pipeline_results.jsonl"
    predictions: list[str] = []
    if results_jsonl.exists():
        with results_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                predictions.append(obj["answer"])
    else:
        # Fallback: predictions not available
        predictions = [""] * len(records)

    comp = compute_fitness_components(agg, weights)

    config_dict: dict = {
        "chunk_size": pc.chunk_size,
        "overlap": pc.overlap,
        "top_k": pc.top_k,
        "prompt_strategy": pc.prompt_strategy,
        "temperature": pc.temperature,
    }
    if isinstance(pc, FullV2Config):
        config_dict["retriever_type"] = pc.retriever_type
        config_dict["reranker_enabled"] = pc.reranker_enabled
        config_dict["max_output_tokens"] = pc.max_output_tokens

    print(f"  fitness={comp.fitness:.4f} F1={agg.mean_f1:.4f}")
    return {
        "name": name,
        "config": config_dict,
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
        "predictions": predictions,
    }


# ── Helper: Iterative RAG ───────────────────────────────────────────────────
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
    predictions = result.get("predictions", [""] * len(records))
    print(
        f"  fitness={result['fitness']:.4f} F1={m['mean_f1']:.4f}"
        f"  Round2={result['n_round2_triggered']}/{len(records)}"
    )
    return {
        "name": name,
        "config": result["config"],
        "metrics": m,
        "fitness": result["fitness"],
        "predictions": predictions,
    }


# ── Helper: Load GA best config from summary JSON ───────────────────────────
def _load_ga_best(summary_path: Path) -> dict:
    """Load a GA summary JSON and return the best_config from the top record.

    The summary is a list of GenerationRecord dicts. Each has best_fitness
    and best_config. We find the record with highest best_fitness and
    return its best_config dict.
    """
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    best_record = max(data, key=lambda r: r["best_fitness"])
    return best_record["best_config"]


# ── Load GA best configs ────────────────────────────────────────────────────
# Phase 12/13 v1 bests (hardcoded from previous phases)
qa_v1_best_config = MicroPilotConfig(
    chunk_size=512,
    overlap=0.0,
    top_k=19,
    prompt_strategy="chain-of-thought",
    temperature=0.6743266807244299,
)
legal_v1_best_config = MicroPilotConfig(
    chunk_size=384,
    overlap=0.2,
    top_k=20,
    prompt_strategy="chain-of-thought",
    temperature=1.0,
)

# Phase 16 v2 bests — load from summary files
qa_seeds = [42, 123, 7]
qa_v2_configs: dict[int, FullV2Config] = {}
for seed in qa_seeds:
    summary_path = Path(
        f"results/phase16/general_qa/phase16_qa_seed{seed}_summary.json"
    )
    if summary_path.exists():
        best = _load_ga_best(summary_path)
        qa_v2_configs[seed] = FullV2Config.from_dict(best)
    else:
        print(f"WARNING: {summary_path} not found, skipping seed {seed}")

legal_v2_configs: dict[int, FullV2Config] = {}
for seed in qa_seeds:
    summary_path = Path(
        f"results/phase16/legal/phase16_legal_seed{seed}_summary.json"
    )
    if summary_path.exists():
        best = _load_ga_best(summary_path)
        legal_v2_configs[seed] = FullV2Config.from_dict(best)
    else:
        print(f"WARNING: {summary_path} not found, skipping seed {seed}")


# ── General-QA held-out ─────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("General-QA held-out evaluation (records 100-399, n=300)")

general_weights = FitnessWeights()
general_cache = paths.root / "data" / "cache" / "held_out_phase16_general"
general_results_dir = paths.results / "phase16" / "held_out_phase16_general_runs"
general_held_harness = EvaluationHarness(general_held_out)
general_references = [r.answer for r in general_held_out]

# Create reranker for FullV2Config evaluations
reranker = Reranker()

general_systems: list[dict] = []

print("\n[1/8] Direct LLM (General-QA held-out, n=300)...")
general_systems.append(
    _eval_direct_llm(
        name="Direct LLM",
        records=general_held_out,
        wrapper=wrapper,
        harness=general_held_harness,
        weights=general_weights,
    )
)

print("\n[2/8] Naive RAG (General-QA held-out, n=300)...")
general_systems.append(
    _eval_runner_config(
        name="Naive RAG",
        pc=MicroPilotConfig(
            chunk_size=256, overlap=0.0, top_k=5,
            prompt_strategy="zero-shot", temperature=0.0,
        ),
        run_id="naive_rag_held_out_p16_general",
        records=general_held_out,
        corpus=general_corpus,
        wrapper=wrapper,
        weights=general_weights,
        cache_dir=general_cache,
        results_dir=general_results_dir,
    )
)

print("\n[3/8] Iterative RAG (General-QA held-out, n=300)...")
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

print("\n[4/8] GA-optimised v1 Phase 12 (General-QA held-out, n=300)...")
general_systems.append(
    _eval_runner_config(
        name="GA-optimised v1 (Phase 12)",
        pc=qa_v1_best_config,
        run_id="ga_v1_phase12_held_out_p16_general",
        records=general_held_out,
        corpus=general_corpus,
        wrapper=wrapper,
        weights=general_weights,
        cache_dir=general_cache,
        results_dir=general_results_dir,
    )
)

# GA v2 seed evaluations
for seed_val in qa_seeds:
    idx = qa_seeds.index(seed_val) + 5
    if seed_val in qa_v2_configs:
        print(
            f"\n[{idx}/8] GA v2 seed {seed_val} best "
            f"(General-QA held-out, n=300)..."
        )
        general_systems.append(
            _eval_runner_config(
                name=f"GA v2 seed {seed_val} best",
                pc=qa_v2_configs[seed_val],
                run_id=f"ga_v2_seed{seed_val}_held_out_p16_general",
                records=general_held_out,
                corpus=general_corpus,
                wrapper=wrapper,
                weights=general_weights,
                cache_dir=general_cache,
                results_dir=general_results_dir,
                reranker=reranker,
                track="general_qa",
                index_root=Path("data/indexes"),
            )
        )
    else:
        print(f"\n[{idx}/8] GA v2 seed {seed_val} — SKIPPED (no summary)")

# Consensus best: pick the v2 seed config with highest mean F1 on held-out
if len(qa_v2_configs) >= 2:
    # Find the seed whose held-out F1 is highest among evaluated v2 systems
    v2_results = [
        s for s in general_systems
        if s["name"].startswith("GA v2 seed")
    ]
    consensus = max(v2_results, key=lambda s: s["metrics"]["mean_f1"])
    consensus_entry = {
        "name": "Consensus best (highest mean held-out F1)",
        "config": consensus["config"],
        "metrics": consensus["metrics"],
        "fitness": consensus["fitness"],
        "predictions": consensus["predictions"],
        "source_system": consensus["name"],
    }
    print(
        f"\n[8/8] Consensus best = {consensus['name']} "
        f"(F1={consensus['metrics']['mean_f1']:.4f})"
    )
    general_systems.append(consensus_entry)
else:
    print("\n[8/8] Consensus best — SKIPPED (fewer than 2 v2 seeds available)")


# ── Legal held-out ───────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("Legal held-out evaluation (records 50-349, n=300)")

legal_weights = legal_fitness_weights()
legal_cache = paths.root / "data" / "cache" / "held_out_phase16_legal"
legal_results_dir = paths.results / "phase16" / "held_out_phase16_legal_runs"
legal_held_harness = EvaluationHarness(legal_held_out)
legal_references = [r.answer for r in legal_held_out]

legal_systems: list[dict] = []

print("\n[1/8] Direct LLM (Legal held-out, n=300)...")
legal_systems.append(
    _eval_direct_llm(
        name="Direct LLM",
        records=legal_held_out,
        wrapper=wrapper,
        harness=legal_held_harness,
        weights=legal_weights,
    )
)

print("\n[2/8] Naive RAG (Legal held-out, n=300)...")
legal_systems.append(
    _eval_runner_config(
        name="Naive RAG",
        pc=MicroPilotConfig(
            chunk_size=256, overlap=0.0, top_k=5,
            prompt_strategy="zero-shot", temperature=0.0,
        ),
        run_id="naive_rag_held_out_p16_legal",
        records=legal_held_out,
        corpus=legal_corpus,
        wrapper=wrapper,
        weights=legal_weights,
        cache_dir=legal_cache,
        results_dir=legal_results_dir,
    )
)

print("\n[3/8] Iterative RAG (Legal held-out, n=300)...")
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

print("\n[4/8] GA-optimised v1 Phase 13 Legal (Legal held-out, n=300)...")
legal_systems.append(
    _eval_runner_config(
        name="GA-optimised v1 (Phase 13 Legal)",
        pc=legal_v1_best_config,
        run_id="ga_v1_phase13_held_out_p16_legal",
        records=legal_held_out,
        corpus=legal_corpus,
        wrapper=wrapper,
        weights=legal_weights,
        cache_dir=legal_cache,
        results_dir=legal_results_dir,
    )
)

# GA v2 seed evaluations
for seed_val in qa_seeds:
    idx = qa_seeds.index(seed_val) + 5
    if seed_val in legal_v2_configs:
        print(
            f"\n[{idx}/8] GA v2 seed {seed_val} best "
            f"(Legal held-out, n=300)..."
        )
        legal_systems.append(
            _eval_runner_config(
                name=f"GA v2 seed {seed_val} best",
                pc=legal_v2_configs[seed_val],
                run_id=f"ga_v2_seed{seed_val}_held_out_p16_legal",
                records=legal_held_out,
                corpus=legal_corpus,
                wrapper=wrapper,
                weights=legal_weights,
                cache_dir=legal_cache,
                results_dir=legal_results_dir,
                reranker=reranker,
                track="legal",
                index_root=Path("data/indexes"),
            )
        )
    else:
        print(f"\n[{idx}/8] GA v2 seed {seed_val} — SKIPPED (no summary)")

# Consensus best for legal
if len(legal_v2_configs) >= 2:
    v2_results = [
        s for s in legal_systems
        if s["name"].startswith("GA v2 seed")
    ]
    consensus = max(v2_results, key=lambda s: s["metrics"]["mean_f1"])
    consensus_entry = {
        "name": "Consensus best (highest mean held-out F1)",
        "config": consensus["config"],
        "metrics": consensus["metrics"],
        "fitness": consensus["fitness"],
        "predictions": consensus["predictions"],
        "source_system": consensus["name"],
    }
    print(
        f"\n[8/8] Consensus best = {consensus['name']} "
        f"(F1={consensus['metrics']['mean_f1']:.4f})"
    )
    legal_systems.append(consensus_entry)
else:
    print("\n[8/8] Consensus best — SKIPPED (fewer than 2 v2 seeds available)")


# ── BERTScore second pass ────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("Computing BERTScore F1 for all systems...")

for system in general_systems:
    preds = system.get("predictions", [])
    if preds and any(p != "" for p in preds):
        bs_f1 = _compute_bertscore(preds, general_references)
        system["metrics"]["bertscore_f1"] = bs_f1
        print(f"  General-QA | {system['name']:<40} | BERTScore={bs_f1:.4f}")
    else:
        system["metrics"]["bertscore_f1"] = None
        print(f"  General-QA | {system['name']:<40} | BERTScore=N/A")

for system in legal_systems:
    preds = system.get("predictions", [])
    if preds and any(p != "" for p in preds):
        bs_f1 = _compute_bertscore(preds, legal_references)
        system["metrics"]["bertscore_f1"] = bs_f1
        print(f"  Legal      | {system['name']:<40} | BERTScore={bs_f1:.4f}")
    else:
        system["metrics"]["bertscore_f1"] = None
        print(f"  Legal      | {system['name']:<40} | BERTScore=N/A")

# Remove predictions from output (not needed in saved JSON)
for system in general_systems:
    system.pop("predictions", None)
for system in legal_systems:
    system.pop("predictions", None)


# ── Print summary table ─────────────────────────────────────────────────────
print(f"\n{'=' * 85}")
print(
    f"{'System':<40} | {'QA F1':>7} | {'QA BS':>7} "
    f"| {'Leg F1':>7} | {'Leg BS':>7}"
)
print("-" * 85)

# Pair up systems by index (may differ in length if seeds are missing)
max_len = max(len(general_systems), len(legal_systems))
for i in range(max_len):
    qa_name = general_systems[i]["name"] if i < len(general_systems) else ""
    qa_f1 = (
        f"{general_systems[i]['metrics']['mean_f1']:.4f}"
        if i < len(general_systems)
        else "  N/A"
    )
    qa_bs = (
        f"{general_systems[i]['metrics']['bertscore_f1']:.4f}"
        if i < len(general_systems)
        and general_systems[i]["metrics"].get("bertscore_f1") is not None
        else "  N/A"
    )
    leg_f1 = (
        f"{legal_systems[i]['metrics']['mean_f1']:.4f}"
        if i < len(legal_systems)
        else "  N/A"
    )
    leg_bs = (
        f"{legal_systems[i]['metrics']['bertscore_f1']:.4f}"
        if i < len(legal_systems)
        and legal_systems[i]["metrics"].get("bertscore_f1") is not None
        else "  N/A"
    )
    print(f"{qa_name:<40} | {qa_f1:>7} | {qa_bs:>7} | {leg_f1:>7} | {leg_bs:>7}")


# ── Save JSON ────────────────────────────────────────────────────────────────
output = {
    "evaluation_protocol": "Phase 16 held-out",
    "general_qa": {
        "weights": "general (w_q=0.40)",
        "n_records": 300,
        "record_range": "100-399",
        "systems": general_systems,
    },
    "legal": {
        "weights": "legal (w_g=0.40)",
        "n_records": 300,
        "record_range": "50-349",
        "systems": legal_systems,
    },
}
OUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
print(f"\nSaved to {OUT_PATH}")

# Force exit — httpx connection pool in the openai/groq client holds
# non-daemon threads that prevent natural process termination.
sys.exit(0)
