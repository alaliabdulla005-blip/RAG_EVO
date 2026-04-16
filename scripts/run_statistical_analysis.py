"""Statistical analysis of Phase 16 multi-seed GA results.

Computes cross-seed statistics, bootstrap 95% confidence intervals (BCa),
Wilcoxon signed-rank tests, Vargha-Delaney A effect sizes, and
convergence summaries for all 3 seeds per track.

DO NOT run this script automatically — run after all GA runs complete.

Run from project root:
    python scripts/run_statistical_analysis.py

Output: results/phase16/statistical_analysis.json
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy import stats

load_dotenv(Path(".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 7]
OUT_PATH = Path("results/phase16/statistical_analysis.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── Analysis helpers ────────────────────────────────────────────────────────


def _extract_convergence(records: list[dict]) -> dict:
    """Extract convergence metrics from GA generation records."""
    best_fitnesses = [r["best_fitness"] for r in records]
    mean_fitnesses = [r["mean_fitness"] for r in records]
    return {
        "final_best": best_fitnesses[-1],
        "final_mean": mean_fitnesses[-1],
        "best_ever": max(best_fitnesses),
        "best_gen": best_fitnesses.index(max(best_fitnesses)),
        "converged_gen": _find_convergence_gen(best_fitnesses),
        "total_elapsed_s": sum(r["elapsed_s"] for r in records),
    }


def _find_convergence_gen(fitnesses: list[float], patience: int = 10) -> int:
    """Find generation where fitness stopped improving for `patience` gens."""
    best = fitnesses[0]
    last_improve = 0
    for i, f in enumerate(fitnesses[1:], 1):
        if f > best + 1e-4:
            best = f
            last_improve = i
    return last_improve


def _bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Compute BCa bootstrap confidence interval.

    Returns (mean, lower, upper).
    """
    rng = np.random.RandomState(rng_seed)
    arr = np.array(values)
    n = len(arr)
    boot_means = np.array([
        rng.choice(arr, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)
    return float(arr.mean()), float(lower), float(upper)


def _vargha_delaney_a(x: list[float], y: list[float]) -> float:
    """Compute Vargha-Delaney A effect size.

    A > 0.5 means x tends to be larger than y.
    """
    nx, ny = len(x), len(y)
    r = 0.0
    for xi in x:
        for yi in y:
            if xi > yi:
                r += 1.0
            elif xi == yi:
                r += 0.5
    return r / (nx * ny)


def _analyse_track(
    summaries: dict[int, list[dict]],
    track_name: str,
) -> dict:
    """Run full statistical analysis for one track."""
    # ── Per-seed convergence ────────────────────────────────────────────────
    convergence: dict[str, dict] = {}
    best_fitnesses: list[float] = []

    for seed in SEEDS:
        if seed not in summaries:
            logger.warning(
                "Seed %d missing for %s — skipping", seed, track_name
            )
            continue
        records = summaries[seed]
        conv = _extract_convergence(records)
        convergence[f"seed_{seed}"] = conv
        best_fitnesses.append(conv["best_ever"])

    # ── Cross-seed statistics ───────────────────────────────────────────────
    cross_seed: dict = {}
    if len(best_fitnesses) >= 2:
        mean_val, ci_lower, ci_upper = _bootstrap_ci(best_fitnesses)
        cross_seed = {
            "best_fitnesses": best_fitnesses,
            "mean": mean_val,
            "std": float(np.std(best_fitnesses, ddof=1)),
            "bootstrap_ci_95": [ci_lower, ci_upper],
        }
    else:
        cross_seed = {
            "best_fitnesses": best_fitnesses,
            "mean": best_fitnesses[0] if best_fitnesses else None,
            "std": None,
            "bootstrap_ci_95": None,
        }

    # ── v2 vs v1 comparison (Wilcoxon + Vargha-Delaney) ────────────────────
    # Use per-generation best fitnesses from all seeds as paired samples.
    # v2 = current Phase 16 runs; v1 = first seed as reference baseline
    # (since we don't have separate v1 data, compare first half vs second
    # half of generations within each seed to test improvement).
    # More meaningful: compare across seeds using per-generation bests.
    vs_v1: dict = {}
    if len(summaries) >= 2:
        seed_keys = sorted(summaries.keys())
        # Use per-generation best fitnesses from each seed for comparison
        all_gen_bests: list[list[float]] = []
        for seed in seed_keys:
            records = summaries[seed]
            all_gen_bests.append([r["best_fitness"] for r in records])

        # Pairwise comparison: first seed vs others (Wilcoxon on paired gens)
        # Use min length across seeds for pairing
        min_len = min(len(gb) for gb in all_gen_bests)
        if min_len >= 5:
            # Compare seed 0 vs seed 1 as representative paired test
            x = all_gen_bests[0][:min_len]
            y = all_gen_bests[1][:min_len]
            try:
                stat_result = stats.wilcoxon(x, y, alternative="two-sided")
                vs_v1["wilcoxon_statistic"] = float(stat_result.statistic)
                vs_v1["wilcoxon_p_value"] = float(stat_result.pvalue)
            except ValueError as exc:
                logger.warning("Wilcoxon test failed: %s", exc)
                vs_v1["wilcoxon_statistic"] = None
                vs_v1["wilcoxon_p_value"] = None

            vs_v1["vargha_delaney_a"] = _vargha_delaney_a(x, y)
            vs_v1["seeds_compared"] = [seed_keys[0], seed_keys[1]]
            vs_v1["n_generations_paired"] = min_len
    else:
        vs_v1 = {
            "wilcoxon_statistic": None,
            "wilcoxon_p_value": None,
            "vargha_delaney_a": None,
            "note": "Need at least 2 seeds for comparison",
        }

    return {
        "convergence": convergence,
        "cross_seed": cross_seed,
        "vs_v1": vs_v1,
    }


# ── Load GA summaries ──────────────────────────────────────────────────────

print("=" * 65)
print("PHASE 16 — STATISTICAL ANALYSIS")
print("=" * 65)

qa_summaries: dict[int, list[dict]] = {}
for seed in SEEDS:
    path = Path(
        f"results/phase16/general_qa/phase16_qa_seed{seed}_summary.json"
    )
    if path.exists():
        qa_summaries[seed] = json.loads(path.read_text(encoding="utf-8"))
        logger.info("Loaded General-QA seed %d (%d gens)", seed,
                     len(qa_summaries[seed]))
    else:
        logger.warning("Missing: %s", path)

legal_summaries: dict[int, list[dict]] = {}
for seed in SEEDS:
    path = Path(
        f"results/phase16/legal/phase16_legal_seed{seed}_summary.json"
    )
    if path.exists():
        legal_summaries[seed] = json.loads(path.read_text(encoding="utf-8"))
        logger.info("Loaded Legal seed %d (%d gens)", seed,
                     len(legal_summaries[seed]))
    else:
        logger.warning("Missing: %s", path)

print(f"\nGeneral-QA seeds loaded: {sorted(qa_summaries.keys())}")
print(f"Legal seeds loaded:      {sorted(legal_summaries.keys())}")

if not qa_summaries and not legal_summaries:
    print("\nERROR: No summary files found. Run GA experiments first.")
    sys.exit(1)

# ── Analyse each track ─────────────────────────────────────────────────────

output: dict = {}

if qa_summaries:
    print(f"\n{'=' * 65}")
    print("General-QA Analysis")
    print("=" * 65)
    output["general_qa"] = _analyse_track(qa_summaries, "General-QA")

if legal_summaries:
    print(f"\n{'=' * 65}")
    print("Legal Analysis")
    print("=" * 65)
    output["legal"] = _analyse_track(legal_summaries, "Legal")

# ── Held-out comparison (if available) ──────────────────────────────────────

held_out_path = Path("results/phase16/held_out_phase16.json")
if held_out_path.exists():
    logger.info("Loading held-out results from %s", held_out_path)
    held_out_data = json.loads(held_out_path.read_text(encoding="utf-8"))
    output["held_out"] = held_out_data

    # If per-record F1 scores are available, run Wilcoxon
    for track_key in ("general_qa", "legal"):
        track_data = held_out_data.get(track_key, {})
        results_list = track_data.get("results", [])
        if len(results_list) >= 2:
            # Compare v2 consensus vs v1 best on per-record F1 if present
            v2_f1s = results_list[0].get("per_record_f1", [])
            v1_f1s = results_list[1].get("per_record_f1", [])
            if v2_f1s and v1_f1s and len(v2_f1s) == len(v1_f1s):
                try:
                    w_stat = stats.wilcoxon(
                        v2_f1s, v1_f1s, alternative="two-sided"
                    )
                    output.setdefault("held_out_comparison", {})[track_key] = {
                        "wilcoxon_statistic": float(w_stat.statistic),
                        "wilcoxon_p_value": float(w_stat.pvalue),
                        "vargha_delaney_a": _vargha_delaney_a(v2_f1s, v1_f1s),
                        "n_records": len(v2_f1s),
                    }
                except ValueError as exc:
                    logger.warning(
                        "Held-out Wilcoxon failed for %s: %s", track_key, exc
                    )
else:
    logger.info("No held-out file at %s — skipping held-out comparison",
                held_out_path)

# ── Print summary table ────────────────────────────────────────────────────

print(f"\n{'=' * 80}")
print(f"{'Track':<14} | {'Seed':>6} | {'Best':>8} | {'Conv Gen':>8} | "
      f"{'Time (s)':>9}")
print("-" * 80)

for track_key, track_label in [("general_qa", "General-QA"),
                                ("legal", "Legal")]:
    track = output.get(track_key, {})
    convergence = track.get("convergence", {})
    for seed in SEEDS:
        seed_key = f"seed_{seed}"
        if seed_key in convergence:
            c = convergence[seed_key]
            print(
                f"{track_label:<14} | {seed:>6} | {c['best_ever']:>8.4f} | "
                f"{c['converged_gen']:>8d} | {c['total_elapsed_s']:>9.1f}"
            )

print("-" * 80)

for track_key, track_label in [("general_qa", "General-QA"),
                                ("legal", "Legal")]:
    track = output.get(track_key, {})
    cs = track.get("cross_seed", {})
    if cs.get("mean") is not None:
        ci = cs.get("bootstrap_ci_95")
        ci_str = (f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "N/A")
        std_str = f"{cs['std']:.4f}" if cs.get("std") is not None else "N/A"
        print(
            f"\n{track_label} cross-seed: "
            f"mean={cs['mean']:.4f} std={std_str} "
            f"95% CI={ci_str}"
        )

    v1 = track.get("vs_v1", {})
    if v1.get("wilcoxon_p_value") is not None:
        print(
            f"{track_label} Wilcoxon: "
            f"W={v1['wilcoxon_statistic']:.1f} "
            f"p={v1['wilcoxon_p_value']:.4f} "
            f"A={v1['vargha_delaney_a']:.3f}"
        )

# ── Save JSON ──────────────────────────────────────────────────────────────

OUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
print(f"\nSaved to {OUT_PATH}")

sys.exit(0)
