"""Phase 16 held-out evaluation — all systems on expanded held-out test sets.

Evaluates 20 total rows (10 per track, asymmetric) on held-out records that
were never accessed during GA search, plus BERTScore F1 as an extended metric.

General-QA bucket (records[100:400], n=300):
  1.  Direct-LLM              (no retrieval)
  2.  Naive-RAG               (chunk=256, top_k=5, zero-shot)
  3.  Conservative-RAG        (chunk=256, top_k=3, zero-shot)
  4.  Iterative-RAG           (2-round adaptive)
  5.  P12-Best                (Phase 12 QA best — native)
  6.  P13-Best-on-QA          (Phase 13 Legal best on QA — cross)
  7.  GA-P16-QA-S42           (Phase 16 QA seed 42 — native)
  8.  GA-P16-QA-S123          (Phase 16 QA seed 123 — native)
  9.  GA-P16-QA-S7            (Phase 16 QA seed 7 — native)
  10. GA-P16-Legal-S42-on-QA  (Phase 16 Legal seed 42 on QA — cross)

Legal bucket (records[50:350], n=300): analogous layout.

Supports resume: re-running after interruption skips any (track, name) already
present in results/phase16/held_out_phase16.json. BERTScore also resumes
per-row on the bertscore_f1 field.

DO NOT run this script automatically — run after GA search completes and the
debug subsets have been rebuilt to >= 500 / >= 400 records via
scripts/rebuild_debug_subsets.py.

Run from project root:
    INFERENCE_PROVIDER=deepinfra python scripts/run_held_out_phase16.py

Output: results/phase16/held_out_phase16.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(".env"))

os.environ["SNAP_CACHE_KEYS"] = "1"
os.environ.setdefault("ASYNC_EVAL", "0")

from rag_evo.cache.cache import ResultCache
from rag_evo.config.schema import FullV2Config, MicroPilotConfig
from rag_evo.data.hotpotqa_loader import load_records
from rag_evo.data.legal_loader import load_legal_records
from rag_evo.evaluation.harness import EvaluationHarness, RecordMetrics
from rag_evo.evaluation.iterative_rag import run_iterative_rag
from rag_evo.ga.fitness import (
    FitnessWeights,
    compute_fitness_components,
    legal_fitness_weights,
)
from rag_evo.generation.config import DecodingConfig
from rag_evo.generation.wrapper import GeneratorWrapper
from rag_evo.pipeline.prompt_builder import SYSTEM_PROMPT_ZERO_SHOT
from rag_evo.pipeline.rag_pipeline import PipelineResult, load_results
from rag_evo.retrieval.reranker import Reranker
from rag_evo.runner.experiment_runner import ExperimentRunner, RunConfig
from rag_evo.utils.paths import get_project_paths

_CITATION_RE = re.compile(r"\[DOC_\d+\]")

paths = get_project_paths()
OUT_PATH = Path("results/phase16/held_out_phase16.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

GENERAL_SLICE = slice(100, 400)
LEGAL_SLICE = slice(50, 350)
N_RECORDS_PER_TRACK = 300

GENERAL_CACHE_DIR = paths.root / "data" / "cache" / "held_out_phase16_general"
LEGAL_CACHE_DIR = paths.root / "data" / "cache" / "held_out_phase16_legal"
GENERAL_RESULTS_DIR = paths.results / "phase16" / "held_out_phase16_general_runs"
LEGAL_RESULTS_DIR = paths.results / "phase16" / "held_out_phase16_legal_runs"

# Approximate prompt sizes for the cost estimator — derived from prior runs.
_AVG_TOKENS_PER_CALL = {"general_qa": 1_446, "legal": 6_963}
_DEEPINFRA_USD_PER_M = 0.02

# Phase 16 summary JSONs — source of truth for GA best configs.
P16_QA_SUMMARY = {
    42: Path("results/phase16/general_qa/phase16_qa_seed42_summary.json"),
    123: Path("results/phase16/general_qa/phase16_qa_seed123_summary.json"),
    7: Path("results/phase16/general_qa/phase16_qa_seed7_summary.json"),
}
P16_LEGAL_SUMMARY = {
    42: Path("results/phase16/legal/phase16_legal_seed42_summary.json"),
}

# Manual baseline configs (shared between tracks).
NAIVE_RAG_CONFIG = MicroPilotConfig(
    chunk_size=256,
    overlap=0.0,
    top_k=5,
    prompt_strategy="zero-shot",
    temperature=0.0,
)
CONSERVATIVE_RAG_CONFIG = MicroPilotConfig(
    chunk_size=256,
    overlap=0.0,
    top_k=3,
    prompt_strategy="zero-shot",
    temperature=0.0,
)

# Phase 12 and Phase 13 GA bests (MicroPilotConfig; max_output_tokens=150 is
# a fixed property of that schema). Values come from the planning prompt.
P12_BEST_CONFIG = MicroPilotConfig(
    chunk_size=512,
    overlap=0.0,
    top_k=19,
    prompt_strategy="chain-of-thought",
    temperature=0.6743266807244299,
)
P13_BEST_CONFIG = MicroPilotConfig(
    chunk_size=384,
    overlap=0.0,
    top_k=20,
    prompt_strategy="zero-shot",
    temperature=1.0,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ── BERTScore ───────────────────────────────────────────────────────────────
def _compute_bertscore(predictions: list[str], references: list[str]) -> float:
    """Compute mean BERTScore F1 using microsoft/deberta-xlarge-mnli."""
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        import subprocess

        print("bert-score not installed — installing now...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install",
             "bert-score", "--break-system-packages"]
        )
        from bert_score import score as bert_score_fn

    _, _, f1 = bert_score_fn(
        predictions,
        references,
        model_type="microsoft/deberta-xlarge-mnli",
        batch_size=16,
        verbose=False,
    )
    return f1.mean().item()


# ── GA summary loading ─────────────────────────────────────────────────────
def _load_ga_best(summary_path: Path) -> dict:
    """Return best_config from the max-fitness generation record."""
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    best_record = max(data, key=lambda r: r["best_fitness"])
    return best_record["best_config"]


def _load_full_v2(summary_path: Path) -> FullV2Config:
    return FullV2Config.from_dict(_load_ga_best(summary_path))


# ── Per-record fitness (approximation) ─────────────────────────────────────
def _per_record_fitness(
    rec: RecordMetrics,
    agg_c_norm: float,
    agg_l_norm: float,
    weights: FitnessWeights,
) -> float:
    """Scalar fitness for a single record.

    Quality (q), grounding (g), retrieval (r) use the record's own metrics.
    Cost (c) and latency (l) reuse the aggregate-level normalised values
    because token/latency budgets are defined per run, not per record.
    """
    return (
        weights.w_q * rec.f1
        + weights.w_g * rec.citation_f1
        + weights.w_r * rec.retrieval_recall_at_k
        - weights.w_c * agg_c_norm
        - weights.w_l * agg_l_norm
    )


def _record_metrics_to_dict(rec: RecordMetrics, fitness: float) -> dict:
    return {
        "record_id": rec.record_id,
        "em": rec.em,
        "f1": rec.f1,
        "citation_precision": rec.citation_precision,
        "citation_recall": rec.citation_recall,
        "citation_f1": rec.citation_f1,
        "retrieval_recall_at_k": rec.retrieval_recall_at_k,
        "input_tokens": rec.input_tokens,
        "output_tokens": rec.output_tokens,
        "latency_ms": rec.latency_ms,
        "fitness": fitness,
    }


# ── Output shape normalisation ─────────────────────────────────────────────
def _finalise_entry(
    name: str,
    config: dict | str,
    per_record_metrics: list[RecordMetrics],
    predictions: list[str],
    agg_c_norm: float,
    agg_l_norm: float,
    aggregate_fitness: float,
    weights: FitnessWeights,
) -> dict:
    """Shape a single system result to the spec'd output schema."""
    per_record_fit = [
        _per_record_fitness(rec, agg_c_norm, agg_l_norm, weights)
        for rec in per_record_metrics
    ]
    per_record_payload = [
        {
            **_record_metrics_to_dict(rec, fit),
            "prediction": pred,
        }
        for rec, fit, pred in zip(per_record_metrics, per_record_fit, predictions)
    ]
    n = len(per_record_metrics)
    if n == 0:
        return {
            "name": name,
            "config": config,
            "n_records": 0,
            "mean_fitness": aggregate_fitness,
            "std_fitness": 0.0,
            "mean_em": 0.0,
            "mean_f1": 0.0,
            "mean_cit_precision": 0.0,
            "mean_cit_recall": 0.0,
            "mean_cit_f1": 0.0,
            "mean_ret_r5": 0.0,
            "mean_bertscore_f1": None,
            "per_record": [],
        }
    return {
        "name": name,
        "config": config,
        "n_records": n,
        "mean_fitness": aggregate_fitness,
        "std_fitness": statistics.pstdev(per_record_fit) if n > 1 else 0.0,
        "mean_em": sum(r.em for r in per_record_metrics) / n,
        "mean_f1": sum(r.f1 for r in per_record_metrics) / n,
        "mean_cit_precision": sum(
            r.citation_precision for r in per_record_metrics
        ) / n,
        "mean_cit_recall": sum(
            r.citation_recall for r in per_record_metrics
        ) / n,
        "mean_cit_f1": sum(r.citation_f1 for r in per_record_metrics) / n,
        "mean_ret_r5": sum(
            r.retrieval_recall_at_k for r in per_record_metrics
        ) / n,
        "mean_bertscore_f1": None,
        "per_record": per_record_payload,
    }


# ── Evaluators (produce a finalised entry) ─────────────────────────────────
def _eval_direct_llm(
    name: str,
    records: list,
    wrapper: GeneratorWrapper,
    weights: FitnessWeights,
) -> dict:
    """No-retrieval baseline via direct wrapper calls."""
    harness = EvaluationHarness(records)
    direct_decoding = DecodingConfig(temperature=0.0, max_tokens=150, seed=42)
    pipeline_results: list[PipelineResult] = []

    for record in records:
        output = wrapper.generate(
            system_prompt=SYSTEM_PROMPT_ZERO_SHOT,
            context_block="[No context provided]",
            query=record.question,
            decoding=direct_decoding,
        )
        cited = _CITATION_RE.findall(output.answer)
        pipeline_results.append(
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

    per_record, agg = harness.evaluate_batch(pipeline_results)
    comp = compute_fitness_components(agg, weights)
    predictions = [r.answer for r in pipeline_results]
    return _finalise_entry(
        name=name,
        config="no retrieval, zero-shot, temperature=0.0, max_tokens=150",
        per_record_metrics=per_record,
        predictions=predictions,
        agg_c_norm=comp.c_norm,
        agg_l_norm=comp.l_norm,
        aggregate_fitness=comp.fitness,
        weights=weights,
    )


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
    reranker: Reranker | None,
    track: str,
    index_root: Path,
) -> dict:
    """Evaluate a MicroPilotConfig or FullV2Config via ExperimentRunner."""
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
    per_record = result.per_record_metrics

    # Recover answers from the runner's saved JSONL (preserves per-record
    # order). Missing file means we fall back to empty predictions.
    results_jsonl = results_dir / run_id / "pipeline_results.jsonl"
    predictions_by_id: dict[str, str] = {}
    if results_jsonl.exists():
        for pr in load_results(results_jsonl):
            predictions_by_id[pr.record_id] = pr.answer
    predictions = [predictions_by_id.get(r.record_id, "") for r in per_record]

    comp = compute_fitness_components(agg, weights)
    config_dict: dict[str, Any] = {
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
    else:
        config_dict["max_output_tokens"] = pc.max_output_tokens

    return _finalise_entry(
        name=name,
        config=config_dict,
        per_record_metrics=per_record,
        predictions=predictions,
        agg_c_norm=comp.c_norm,
        agg_l_norm=comp.l_norm,
        aggregate_fitness=comp.fitness,
        weights=weights,
    )


def _evaluate_jsonl(
    jsonl: Path, harness: EvaluationHarness
) -> list[RecordMetrics]:
    """Reconstruct RecordMetrics from a saved pipeline_results.jsonl."""
    if not jsonl.exists():
        return []
    pipeline_results = load_results(jsonl)
    per_record, _ = harness.evaluate_batch(pipeline_results)
    return per_record


def _eval_iterative_rag(
    name: str,
    records: list,
    corpus: list,
    wrapper: GeneratorWrapper,
    weights: FitnessWeights,
    cache_dir: Path,
) -> dict:
    """2-round adaptive Iterative-RAG baseline."""
    harness = EvaluationHarness(records)
    result = run_iterative_rag(
        qa_records=records,
        corpus_documents=corpus,
        config={},
        wrapper=wrapper,
        harness=harness,
        cache_dir=cache_dir,
        fitness_weights=weights,
    )
    return _finalise_entry(
        name=name,
        config=result["config"],
        per_record_metrics=result["per_record_metrics"],
        predictions=result["predictions"],
        agg_c_norm=result["components"]["c_norm"],
        agg_l_norm=result["components"]["l_norm"],
        aggregate_fitness=result["fitness"],
        weights=weights,
    )


# ── System descriptors ─────────────────────────────────────────────────────
def build_all_systems() -> list[dict]:
    """Return the full list of 20 (track, system) descriptors.

    Each entry is a plain dict usable by the main driver to dispatch to
    the appropriate evaluator. Does not require network, wrapper, or
    record loading — safe to call from a verification smoke test.
    """
    qa_s42 = _load_full_v2(P16_QA_SUMMARY[42])
    qa_s123 = _load_full_v2(P16_QA_SUMMARY[123])
    qa_s7 = _load_full_v2(P16_QA_SUMMARY[7])
    legal_s42 = _load_full_v2(P16_LEGAL_SUMMARY[42])

    general_qa: list[dict] = [
        {"track": "general_qa", "name": "Direct-LLM", "kind": "direct"},
        {"track": "general_qa", "name": "Naive-RAG", "kind": "runner",
         "pc": NAIVE_RAG_CONFIG, "run_id": "naive_rag_held_out_general"},
        {"track": "general_qa", "name": "Conservative-RAG", "kind": "runner",
         "pc": CONSERVATIVE_RAG_CONFIG,
         "run_id": "conservative_rag_held_out_general"},
        {"track": "general_qa", "name": "Iterative-RAG", "kind": "iterative"},
        {"track": "general_qa", "name": "P12-Best", "kind": "runner",
         "pc": P12_BEST_CONFIG, "run_id": "p12_best_held_out_general"},
        {"track": "general_qa", "name": "P13-Best-on-QA", "kind": "runner",
         "pc": P13_BEST_CONFIG, "run_id": "p13_best_on_qa_held_out_general"},
        {"track": "general_qa", "name": "GA-P16-QA-S42", "kind": "runner",
         "pc": qa_s42, "run_id": "ga_p16_qa_s42_held_out_general"},
        {"track": "general_qa", "name": "GA-P16-QA-S123", "kind": "runner",
         "pc": qa_s123, "run_id": "ga_p16_qa_s123_held_out_general"},
        {"track": "general_qa", "name": "GA-P16-QA-S7", "kind": "runner",
         "pc": qa_s7, "run_id": "ga_p16_qa_s7_held_out_general"},
        {"track": "general_qa", "name": "GA-P16-Legal-S42-on-QA",
         "kind": "runner", "pc": legal_s42,
         "run_id": "ga_p16_legal_s42_on_qa_held_out_general"},
    ]
    legal: list[dict] = [
        {"track": "legal", "name": "Direct-LLM", "kind": "direct"},
        {"track": "legal", "name": "Naive-RAG", "kind": "runner",
         "pc": NAIVE_RAG_CONFIG, "run_id": "naive_rag_held_out_legal"},
        {"track": "legal", "name": "Conservative-RAG", "kind": "runner",
         "pc": CONSERVATIVE_RAG_CONFIG,
         "run_id": "conservative_rag_held_out_legal"},
        {"track": "legal", "name": "Iterative-RAG", "kind": "iterative"},
        {"track": "legal", "name": "P13-Best", "kind": "runner",
         "pc": P13_BEST_CONFIG, "run_id": "p13_best_held_out_legal"},
        {"track": "legal", "name": "P12-Best-on-Legal", "kind": "runner",
         "pc": P12_BEST_CONFIG, "run_id": "p12_best_on_legal_held_out_legal"},
        {"track": "legal", "name": "GA-P16-Legal-S42", "kind": "runner",
         "pc": legal_s42, "run_id": "ga_p16_legal_s42_held_out_legal"},
        {"track": "legal", "name": "GA-P16-QA-S42-on-Legal", "kind": "runner",
         "pc": qa_s42, "run_id": "ga_p16_qa_s42_on_legal_held_out_legal"},
        {"track": "legal", "name": "GA-P16-QA-S123-on-Legal", "kind": "runner",
         "pc": qa_s123, "run_id": "ga_p16_qa_s123_on_legal_held_out_legal"},
        {"track": "legal", "name": "GA-P16-QA-S7-on-Legal", "kind": "runner",
         "pc": qa_s7, "run_id": "ga_p16_qa_s7_on_legal_held_out_legal"},
    ]
    return general_qa + legal


# ── Output file I/O ────────────────────────────────────────────────────────
def _load_existing_output() -> dict:
    """Load an existing partial output file, or create a fresh skeleton."""
    if OUT_PATH.exists():
        try:
            data = json.loads(OUT_PATH.read_text(encoding="utf-8"))
            if "results" in data and "metadata" in data:
                return data
        except (json.JSONDecodeError, OSError):
            logger.warning("Existing output unreadable — starting fresh")
    from datetime import date
    return {
        "metadata": {
            "n_records_per_track": N_RECORDS_PER_TRACK,
            "provider": os.environ.get("INFERENCE_PROVIDER", "deepinfra"),
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "date": date.today().isoformat(),
            "tracks": ["general_qa", "legal"],
        },
        "results": {"general_qa": [], "legal": []},
    }


def _save_atomic(data: dict) -> None:
    """Atomic write: tempfile + rename."""
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(OUT_PATH.parent),
        prefix=".held_out_phase16.",
        suffix=".tmp",
        delete=False,
    ) as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        tmp_name = f.name
    Path(tmp_name).replace(OUT_PATH)


def _upsert(
    data: dict, track: str, entry: dict
) -> None:
    bucket = data["results"][track]
    for i, row in enumerate(bucket):
        if row["name"] == entry["name"]:
            bucket[i] = entry
            return
    bucket.append(entry)


def _has_result(data: dict, track: str, name: str) -> bool:
    return any(row["name"] == name for row in data["results"].get(track, []))


def _has_bertscore(data: dict, track: str, name: str) -> bool:
    for row in data["results"].get(track, []):
        if row["name"] == name:
            return row.get("mean_bertscore_f1") is not None
    return False


# ── Cost estimate ──────────────────────────────────────────────────────────
def _print_cost_estimate(systems: list[dict]) -> None:
    qa_count = sum(1 for s in systems if s["track"] == "general_qa")
    legal_count = sum(1 for s in systems if s["track"] == "legal")
    qa_tokens = qa_count * N_RECORDS_PER_TRACK * _AVG_TOKENS_PER_CALL["general_qa"]
    legal_tokens = (
        legal_count * N_RECORDS_PER_TRACK * _AVG_TOKENS_PER_CALL["legal"]
    )
    qa_cost = qa_tokens / 1_000_000 * _DEEPINFRA_USD_PER_M
    legal_cost = legal_tokens / 1_000_000 * _DEEPINFRA_USD_PER_M
    total = qa_cost + legal_cost
    print("Estimated cost (DeepInfra FP8 @ $0.02 / 1M tokens):")
    print(
        f"  General-QA: {qa_count} systems x 300 records x "
        f"~{_AVG_TOKENS_PER_CALL['general_qa']:,} tokens = "
        f"~{qa_tokens / 1_000_000:.2f}M tokens -> ${qa_cost:.3f}"
    )
    print(
        f"  Legal:      {legal_count} systems x 300 records x "
        f"~{_AVG_TOKENS_PER_CALL['legal']:,} tokens = "
        f"~{legal_tokens / 1_000_000:.2f}M tokens -> ${legal_cost:.3f}"
    )
    print(f"  Total: ~${total:.3f}")
    if total > 15:
        print(
            "  WARNING: estimate exceeds $15 — consider dropping "
            "cross-track rows to stay under the $17 DeepInfra budget."
        )


# ── Dispatch a single system ───────────────────────────────────────────────
def _dispatch(
    desc: dict,
    records_by_track: dict,
    corpus_by_track: dict,
    weights_by_track: dict,
    cache_by_track: dict,
    results_dir_by_track: dict,
    wrapper: GeneratorWrapper,
    reranker: Reranker,
) -> dict:
    track = desc["track"]
    records = records_by_track[track]
    weights = weights_by_track[track]
    if desc["kind"] == "direct":
        return _eval_direct_llm(
            name=desc["name"],
            records=records,
            wrapper=wrapper,
            weights=weights,
        )
    if desc["kind"] == "iterative":
        return _eval_iterative_rag(
            name=desc["name"],
            records=records,
            corpus=corpus_by_track[track],
            wrapper=wrapper,
            weights=weights,
            cache_dir=cache_by_track[track] / "iterative",
        )
    if desc["kind"] == "runner":
        return _eval_runner_config(
            name=desc["name"],
            pc=desc["pc"],
            run_id=desc["run_id"],
            records=records,
            corpus=corpus_by_track[track],
            wrapper=wrapper,
            weights=weights,
            cache_dir=cache_by_track[track],
            results_dir=results_dir_by_track[track],
            reranker=reranker,
            track=track,
            index_root=Path("data/indexes"),
        )
    raise ValueError(f"Unknown descriptor kind: {desc['kind']}")


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> int:
    # ── Load held-out slices ──
    print("Loading expanded debug subsets...")
    general_records, general_corpus = load_records(
        Path("data/processed/hotpotqa_debug")
    )
    legal_records, legal_corpus = load_legal_records(
        Path("data/processed/legal_debug")
    )
    if len(general_records) < 400:
        raise AssertionError(
            f"Expected >= 400 General-QA records, got {len(general_records)}. "
            "Run scripts/rebuild_debug_subsets.py first."
        )
    if len(legal_records) < 350:
        raise AssertionError(
            f"Expected >= 350 Legal records, got {len(legal_records)}. "
            "Run scripts/rebuild_debug_subsets.py first."
        )
    general_held_out = general_records[GENERAL_SLICE]
    legal_held_out = legal_records[LEGAL_SLICE]
    assert len(general_held_out) == N_RECORDS_PER_TRACK
    assert len(legal_held_out) == N_RECORDS_PER_TRACK

    print(
        f"General-QA: {len(general_records)} total, "
        f"{len(general_held_out)} held-out"
    )
    print(f"Legal:     {len(legal_records)} total, {len(legal_held_out)} held-out")

    # ── Build descriptors + cost estimate ──
    systems = build_all_systems()
    print(f"Configured {len(systems)} evaluations across 2 tracks.")
    _print_cost_estimate(systems)

    # ── Shared resources ──
    wrapper = GeneratorWrapper()
    reranker = Reranker()
    records_by_track = {
        "general_qa": general_held_out,
        "legal": legal_held_out,
    }
    corpus_by_track = {
        "general_qa": general_corpus,
        "legal": legal_corpus,
    }
    weights_by_track = {
        "general_qa": FitnessWeights(),
        "legal": legal_fitness_weights(),
    }
    cache_by_track = {
        "general_qa": GENERAL_CACHE_DIR,
        "legal": LEGAL_CACHE_DIR,
    }
    results_dir_by_track = {
        "general_qa": GENERAL_RESULTS_DIR,
        "legal": LEGAL_RESULTS_DIR,
    }

    # ── Load-or-initialise output file (resume point) ──
    output = _load_existing_output()
    resumed = sum(len(v) for v in output["results"].values())
    if resumed:
        print(f"Resume: {resumed} entries already present, will skip those.")

    # ── Pass 1: evaluations ──
    for i, desc in enumerate(systems, start=1):
        track = desc["track"]
        name = desc["name"]
        if _has_result(output, track, name):
            print(f"[{i}/{len(systems)}] {track} | {name} — SKIP (done)")
            continue
        print(f"[{i}/{len(systems)}] {track} | {name} — running...")
        entry = _dispatch(
            desc,
            records_by_track,
            corpus_by_track,
            weights_by_track,
            cache_by_track,
            results_dir_by_track,
            wrapper,
            reranker,
        )
        _upsert(output, track, entry)
        _save_atomic(output)
        print(
            f"Completed {name} on {track}: "
            f"fitness={entry['mean_fitness']:.4f} "
            f"EM={entry['mean_em']:.4f} "
            f"F1={entry['mean_f1']:.4f} "
            f"CitF1={entry['mean_cit_f1']:.4f} "
            f"RetR5={entry['mean_ret_r5']:.4f} "
            f"({entry['n_records']} records)"
        )

    # ── Pass 2: BERTScore ──
    references_by_track = {
        "general_qa": [r.gold_answer for r in general_held_out],
        "legal": [r.gold_answer for r in legal_held_out],
    }
    print("\n" + "=" * 70)
    print("BERTScore F1 pass (microsoft/deberta-xlarge-mnli, batch_size=16)")
    for track in ("general_qa", "legal"):
        for row in output["results"][track]:
            if row.get("mean_bertscore_f1") is not None:
                continue
            preds = [pr["prediction"] for pr in row["per_record"]]
            if not preds or not any(p for p in preds):
                row["mean_bertscore_f1"] = None
                continue
            refs = references_by_track[track][: len(preds)]
            bs_f1 = _compute_bertscore(preds, refs)
            row["mean_bertscore_f1"] = bs_f1
            _save_atomic(output)
            print(f"  {track:<10} | {row['name']:<28} | BERTScore F1={bs_f1:.4f}")

    # ── Final summary table ──
    print("\n" + "=" * 85)
    print(f"{'System':<30} | {'Track':<10} | "
          f"{'F1':>7} | {'CitF1':>7} | {'BS F1':>7} | {'Fitness':>8}")
    print("-" * 85)
    for track in ("general_qa", "legal"):
        for row in output["results"][track]:
            bs = row.get("mean_bertscore_f1")
            bs_str = f"{bs:.4f}" if bs is not None else "   N/A"
            print(
                f"{row['name']:<30} | {track:<10} | "
                f"{row['mean_f1']:>7.4f} | {row['mean_cit_f1']:>7.4f} | "
                f"{bs_str:>7} | {row['mean_fitness']:>8.4f}"
            )

    _save_atomic(output)
    print(f"\nSaved to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
