"""Temperature sensitivity ablation — legal track (Phase 15, Task 1).

Investigates whether temperature=1.0 is genuinely optimal for the legal
track or a noise artefact from stochastic evaluation on 50 records.
Evaluates Phase 13 best config at 5 temperatures on ALL 150 legal records.

See limitations key in output JSON: temperature=1.0 may be a noise artefact.

Fixed genes: chunk_size=384, overlap=0.2, top_k=20, CoT.
Temperature sweep: [0.0, 0.3, 0.5, 0.7, 1.0]

Run from project root:
    INFERENCE_PROVIDER=sambanova python scripts/run_temperature_ablation_legal.py

DO NOT run this script — run it manually.
Requires INFERENCE_PROVIDER and matching API key in .env.
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv(Path(".env"))

from rag_evo.cache.cache import ResultCache
from rag_evo.config.schema import MicroPilotConfig
from rag_evo.data.legal_loader import load_legal_records
from rag_evo.evaluation.harness import EvaluationHarness
from rag_evo.ga.fitness import compute_fitness_components, legal_fitness_weights
from rag_evo.generation.wrapper import GeneratorWrapper
from rag_evo.runner.experiment_runner import ExperimentRunner, RunConfig
from rag_evo.utils.paths import get_project_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

TEMPERATURES = [0.0, 0.3, 0.5, 0.7, 1.0]

# Phase 13 best config — only temperature varies
BASE_CHUNK_SIZE = 384
BASE_OVERLAP = 0.2
BASE_TOP_K = 20
BASE_PROMPT_STRATEGY = "chain-of-thought"

# Dev set fitness at temperature=1.0 from legal_baselines.json (50 records)
DEV_FITNESS = 0.185

OUT_JSON = Path("results/ablations/temperature_ablation_legal.json")
OUT_FIG = Path("results/figures/temperature_ablation_legal.png")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

LIMITATIONS = [
    (
        "Small development set (50 records) used for both GA search and final"
        " evaluation — no held-out test set during optimisation."
        " Evaluation noise CV~0.087."
    ),
    (
        "Temperature=1.0 for legal best config is potentially a noise artefact"
        " from stochastic evaluation on 50 records."
        " Temperature ablation (Task 1) investigates this."
    ),
    (
        "BM25-only retrieval throughout v1. Dense and hybrid retrieval are not"
        " explored. Retrieval recall on General-QA saturates at 0.730"
        " — a BM25 ceiling."
    ),
    (
        "Citation format mismatch on legal data: CUAD answers do not naturally"
        " elicit [DOC_N] citation markers, causing CitF1=0.123 despite"
        " RetR@5=0.940."
    ),
    (
        "Legal track GA interrupted by SambaNova 20M/day quota exhaustion."
        " Resumed on new account — results valid but infrastructure instability"
        " documented."
    ),
    (
        "5-gene search space only (chunk_size, overlap, top_k, prompt_strategy,"
        " temperature). retriever_type, reranker_enabled, max_output_tokens not"
        " searched in v1."
    ),
]

paths = get_project_paths()

print("Loading legal debug subset (150 records)...")
legal_records, legal_corpus = load_legal_records(
    Path("data/processed/legal_debug")
)
print(f"Loaded {len(legal_records)} records, {len(legal_corpus)} corpus docs")

wrapper = GeneratorWrapper()
weights = legal_fitness_weights()
harness = EvaluationHarness(legal_records)

ablation_results = []

for temp in TEMPERATURES:
    print(f"\n{'=' * 55}")
    temp_tag = str(temp).replace(".", "_")
    print(f"Temperature: {temp}")

    pc = MicroPilotConfig(
        chunk_size=BASE_CHUNK_SIZE,
        overlap=BASE_OVERLAP,
        top_k=BASE_TOP_K,
        prompt_strategy=BASE_PROMPT_STRATEGY,
        temperature=temp,
    )

    cache = ResultCache(
        paths.root / "data" / "cache" / f"temp_ablation_legal_{temp_tag}"
    )
    runner = ExperimentRunner(wrapper=wrapper, harness=harness, cache=cache)

    run_config = RunConfig(
        pipeline_config=pc,
        run_id=f"temp_ablation_{temp_tag}",
        qa_records=legal_records,
        corpus_documents=legal_corpus,
        cache_dir=paths.root / "data" / "cache" / f"temp_ablation_legal_{temp_tag}",
        results_dir=paths.results / "ablations" / "temperature_ablation_runs",
        model_id=wrapper._model_id,
        seed=42,
    )

    result = runner.run(run_config)
    agg = result.aggregate_metrics
    comp = compute_fitness_components(agg, weights)

    vs_dev = "higher" if comp.fitness > DEV_FITNESS else "lower"
    print(
        f"  fitness={comp.fitness:.4f} ({vs_dev} than dev)"
        f"  F1={agg.mean_f1:.4f}"
        f"  CitF1={agg.mean_citation_f1:.4f}"
        f"  RetR@k={agg.mean_retrieval_recall_at_k:.4f}"
    )

    ablation_results.append({
        "temperature": temp,
        "fitness": comp.fitness,
        "f1": agg.mean_f1,
        "cit_f1": agg.mean_citation_f1,
        "ret_r_at_k": agg.mean_retrieval_recall_at_k,
        "cit_p": agg.mean_citation_precision,
        "cit_r": agg.mean_citation_recall,
        "mean_input_tokens": agg.mean_input_tokens,
        "mean_latency_ms": agg.mean_latency_ms,
        "vs_dev": vs_dev,
    })

# ── Print summary table ───────────────────────────────────────────────────────
print(f"\n{'=' * 75}")
print(
    f"{'Temp':>5} | {'Fitness':>8} | {'F1':>6} | {'CitF1':>6} "
    f"| {'RetR@5':>6} | {'CitP':>6} | {'CitR':>6}"
)
print("-" * 75)
for r in ablation_results:
    print(
        f"{r['temperature']:>5.1f} | {r['fitness']:>8.4f} | {r['f1']:>6.4f}"
        f" | {r['cit_f1']:>6.4f} | {r['ret_r_at_k']:>6.4f}"
        f" | {r['cit_p']:>6.4f} | {r['cit_r']:>6.4f}"
    )
print(f"\nDev set fitness (n=50, temp=1.0): {DEV_FITNESS:.3f}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
summary = {
    "ablation": "temperature",
    "track": "legal",
    "base_config": {
        "chunk_size": BASE_CHUNK_SIZE,
        "overlap": BASE_OVERLAP,
        "top_k": BASE_TOP_K,
        "prompt_strategy": BASE_PROMPT_STRATEGY,
    },
    "n_records": len(legal_records),
    "dev_fitness_at_1_0": DEV_FITNESS,
    "limitations": LIMITATIONS,
    "results": ablation_results,
}
OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"\nSaved to {OUT_JSON}")

# ── Matplotlib figure (dual y-axis) ───────────────────────────────────────────
temps = [r["temperature"] for r in ablation_results]
fitnesses = [r["fitness"] for r in ablation_results]
cit_f1s = [r["cit_f1"] for r in ablation_results]

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(temps, fitnesses, color="blue", linewidth=2, marker="o", label="Fitness")
ax1.axhline(
    y=DEV_FITNESS,
    color="red",
    linestyle="--",
    linewidth=1.2,
    label=f"Phase 13 dev set result ({DEV_FITNESS})",
)
ax1.set_xlabel("Temperature")
ax1.set_ylabel("Fitness", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(
    temps, cit_f1s, color="orange", linewidth=1.5,
    linestyle="--", marker="s", label="Citation F1",
)
ax2.set_ylabel("Citation F1", color="orange")
ax2.tick_params(axis="y", labelcolor="orange")

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

ax1.set_title("Legal Track Temperature Sensitivity (n=150)")
ax1.grid(True, alpha=0.3)
ax1.set_xticks(TEMPERATURES)

fig.tight_layout()
fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print(f"Figure saved to {OUT_FIG}")

# Force exit — httpx connection pool in openai/groq client holds
# non-daemon threads that prevent natural process termination.
sys.exit(0)
