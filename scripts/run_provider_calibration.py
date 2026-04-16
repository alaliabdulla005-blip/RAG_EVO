"""Provider calibration — DeepInfra vs SambaNova on Phase 12 best config.

Evaluates the Phase 12 GA-optimised config (chunk=512, overlap=0.0,
top_k=19, CoT, temp=0.674) on the first 50 General-QA dev records
using DeepInfra. Compares results against the SambaNova evaluation
of the same config on the same dev records (from Phase 12 run
artifacts) to verify provider equivalence within ±3% tolerance.

DO NOT run this script automatically — run manually after DeepInfra
API key is configured.

Run from project root:
    INFERENCE_PROVIDER=deepinfra python scripts/run_provider_calibration.py

Output: results/phase16/calibration.json
"""

import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

from rag_evo.cache.cache import ResultCache
from rag_evo.config.schema import MicroPilotConfig
from rag_evo.data.hotpotqa_loader import load_records
from rag_evo.evaluation.harness import AggregateMetrics, EvaluationHarness
from rag_evo.ga.fitness import FitnessWeights, compute_fitness_components
from rag_evo.generation.wrapper import GeneratorWrapper
from rag_evo.runner.experiment_runner import ExperimentRunner, RunConfig
from rag_evo.utils.paths import get_project_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

paths = get_project_paths()

OUT_PATH = Path("results/phase16/calibration.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

TOLERANCE_PCT = 3.0

# ── Load HotpotQA debug subset, first 50 records ───────────────────────────
print("Loading HotpotQA debug subset...")
all_records, corpus_documents = load_records(
    Path("data/processed/hotpotqa_debug")
)
records = all_records[:50]
print(f"Using {len(records)} records, {len(corpus_documents)} corpus docs")

# ── Phase 12 best config ────────────────────────────────────────────────────
phase12_best = MicroPilotConfig(
    chunk_size=512,
    overlap=0.0,
    top_k=19,
    prompt_strategy="chain-of-thought",
    temperature=0.6743266807244299,
)

# ── Evaluate on DeepInfra ───────────────────────────────────────────────────
print("\nEvaluating Phase 12 best config on DeepInfra (n=50)...")
wrapper = GeneratorWrapper()
harness = EvaluationHarness(records)
cache = ResultCache(paths.root / "data" / "cache" / "provider_calibration")
runner = ExperimentRunner(wrapper=wrapper, harness=harness, cache=cache)

weights = FitnessWeights()

run_config = RunConfig(
    pipeline_config=phase12_best,
    run_id="provider_calibration_deepinfra",
    qa_records=records,
    corpus_documents=corpus_documents,
    cache_dir=paths.root / "data" / "cache" / "provider_calibration",
    results_dir=paths.results / "phase16" / "calibration_runs",
    model_id=wrapper._model_id,
    seed=42,
)
result = runner.run(run_config)
agg = result.aggregate_metrics
comp = compute_fitness_components(agg, weights)

deepinfra_metrics = {
    "mean_em": agg.mean_em,
    "mean_f1": agg.mean_f1,
    "mean_citation_precision": agg.mean_citation_precision,
    "mean_citation_recall": agg.mean_citation_recall,
    "mean_citation_f1": agg.mean_citation_f1,
    "mean_retrieval_recall_at_k": agg.mean_retrieval_recall_at_k,
    "mean_input_tokens": agg.mean_input_tokens,
    "mean_latency_ms": agg.mean_latency_ms,
}
deepinfra_fitness = comp.fitness

print(f"  DeepInfra fitness={deepinfra_fitness:.4f} F1={agg.mean_f1:.4f}")

# ── Load SambaNova reference from Phase 12 run artifacts ───────────────────
print("\nSearching Phase 12 run artifacts for SambaNova reference (dev 0-49)...")
phase12_runs_dir = Path("results/phase12/general_qa/runs")
ref_config = phase12_best.to_dict()
# Remove fixed constants for comparison
ref_config.pop("citations_required", None)
ref_config.pop("top_p", None)

samba_agg_path = None
for run_dir in sorted(phase12_runs_dir.iterdir(), reverse=True):
    config_path = run_dir / "config.json"
    if not config_path.exists():
        continue
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    saved.pop("citations_required", None)
    saved.pop("top_p", None)
    if saved == ref_config:
        samba_agg_path = run_dir / "aggregate_metrics.json"
        print(f"  Found matching run: {run_dir.name}")
        break

if samba_agg_path is None or not samba_agg_path.exists():
    print("ERROR: No Phase 12 run found with the best config on dev records 0-49.")
    print("Expected runs in results/phase12/general_qa/runs/")
    sys.exit(1)

samba_agg = json.loads(samba_agg_path.read_text(encoding="utf-8"))
assert samba_agg["num_records"] == 50, (
    f"Expected 50 records in SambaNova reference, got {samba_agg['num_records']}"
)

# Build metrics dict in the same format as deepinfra_metrics
samba_metrics = {
    "mean_em": samba_agg["mean_em"],
    "mean_f1": samba_agg["mean_f1"],
    "mean_citation_precision": samba_agg["mean_citation_precision"],
    "mean_citation_recall": samba_agg["mean_citation_recall"],
    "mean_citation_f1": samba_agg["mean_citation_f1"],
    "mean_retrieval_recall_at_k": samba_agg["mean_retrieval_recall_at_k"],
    "mean_input_tokens": samba_agg["mean_input_tokens"],
    "mean_latency_ms": samba_agg["mean_latency_ms"],
}

# Compute SambaNova fitness from aggregate metrics
samba_agg_obj = AggregateMetrics(**samba_agg)
samba_comp = compute_fitness_components(samba_agg_obj, weights)
samba_fitness = samba_comp.fitness

print(f"  SambaNova fitness={samba_fitness:.4f} F1={samba_metrics['mean_f1']:.4f}")

# ── Compute delta table ─────────────────────────────────────────────────────
COMPARISON_METRICS = [
    "mean_em",
    "mean_f1",
    "mean_citation_precision",
    "mean_citation_recall",
    "mean_citation_f1",
    "mean_retrieval_recall_at_k",
    "mean_input_tokens",
    "mean_latency_ms",
]

deltas: list[dict] = []
for metric in COMPARISON_METRICS:
    di_val = deepinfra_metrics[metric]
    sn_val = samba_metrics[metric]
    abs_delta = di_val - sn_val
    if sn_val != 0:
        pct_delta = (abs_delta / sn_val) * 100.0
    else:
        pct_delta = 0.0 if abs_delta == 0 else float("inf")
    deltas.append({
        "metric": metric,
        "deepinfra": di_val,
        "sambanova": sn_val,
        "abs_delta": abs_delta,
        "pct_delta": pct_delta,
    })

# ── Verdict ─────────────────────────────────────────────────────────────────
f1_delta = next(d for d in deltas if d["metric"] == "mean_f1")
cit_f1_delta = next(d for d in deltas if d["metric"] == "mean_citation_f1")

f1_pass = abs(f1_delta["pct_delta"]) <= TOLERANCE_PCT
cit_f1_pass = abs(cit_f1_delta["pct_delta"]) <= TOLERANCE_PCT
verdict = "PASS" if (f1_pass and cit_f1_pass) else "FAIL"

# ── Save JSON ───────────────────────────────────────────────────────────────
output = {
    "phase": "Phase 16 — Provider calibration",
    "config": {
        "chunk_size": phase12_best.chunk_size,
        "overlap": phase12_best.overlap,
        "top_k": phase12_best.top_k,
        "prompt_strategy": phase12_best.prompt_strategy,
        "temperature": phase12_best.temperature,
    },
    "n_records": len(records),
    "tolerance_pct": TOLERANCE_PCT,
    "deepinfra": {
        "metrics": deepinfra_metrics,
        "fitness": deepinfra_fitness,
    },
    "sambanova": {
        "source": str(samba_agg_path),
        "metrics": samba_metrics,
        "fitness": samba_fitness,
    },
    "deltas": deltas,
    "verdict": verdict,
    "f1_pass": f1_pass,
    "citation_f1_pass": cit_f1_pass,
}
OUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
print(f"\nSaved to {OUT_PATH}")

# ── Print summary table ─────────────────────────────────────────────────────
print(f"\n{'=' * 78}")
print(f"{'Metric':<30} | {'DeepInfra':>10} | {'SambaNova':>10} | "
      f"{'Delta':>8} | {'%':>7}")
print("-" * 78)
for d in deltas:
    print(
        f"{d['metric']:<30} | {d['deepinfra']:>10.4f} | "
        f"{d['sambanova']:>10.4f} | {d['abs_delta']:>+8.4f} | "
        f"{d['pct_delta']:>+7.2f}%"
    )
print("-" * 78)
print(f"\nVerdict: {verdict}")
print(f"  mean_f1 delta:        {f1_delta['pct_delta']:+.2f}% "
      f"({'PASS' if f1_pass else 'FAIL'}, threshold +/-{TOLERANCE_PCT}%)")
print(f"  mean_citation_f1 delta: {cit_f1_delta['pct_delta']:+.2f}% "
      f"({'PASS' if cit_f1_pass else 'FAIL'}, threshold +/-{TOLERANCE_PCT}%)")

# Force exit — httpx connection pool in the openai/groq client holds
# non-daemon threads that prevent natural process termination.
sys.exit(0)
