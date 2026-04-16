"""Generate Phase 16 dissertation figures.

Produces 5 figures from Phase 16 results:
  1. Multi-seed convergence — General-QA (3 seeds on one plot)
  2. Multi-seed convergence — Legal (3 seeds on one plot)
  3. System comparison bar chart — General-QA held-out
  4. System comparison bar chart — Legal held-out
  5. Extended metrics (BERTScore + RAGAS Faithfulness)

DO NOT run this script automatically — run after all evaluations complete.

Run from project root:
    python scripts/generate_phase16_figures.py

Output: results/phase16/figures/
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ── Output directory ────────────────────────────────────────────────────────
OUT_DIR = Path("results/phase16/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Matplotlib style ────────────────────────────────────────────────────────
plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 150

SEEDS = [42, 123, 7]


# ── Helpers ─────────────────────────────────────────────────────────────────
def _try_load(path: Path) -> dict | list | None:
    """Load a JSON file, returning None with a warning if missing."""
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping dependent figure(s)")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_convergence(
    summaries: dict[int, list[dict]], title: str, out_path: Path
) -> None:
    """Plot multi-seed convergence curves (best + mean) on a single figure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {42: "#2196F3", 123: "#FF9800", 7: "#4CAF50"}
    for seed, records in summaries.items():
        gens = [r["generation_idx"] for r in records]
        best = [r["best_fitness"] for r in records]
        mean = [r["mean_fitness"] for r in records]
        ax.plot(
            gens, best,
            color=colors[seed], linewidth=2,
            label=f"Seed {seed} (best)",
        )
        ax.plot(
            gens, mean,
            color=colors[seed], linewidth=1, linestyle="--", alpha=0.6,
            label=f"Seed {seed} (mean)",
        )
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fitness", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Figure 1 — Multi-seed convergence: General-QA ──────────────────────────
print("Figure 1 — Multi-seed convergence: General-QA")

qa_summaries: dict[int, list[dict]] = {}
for seed in SEEDS:
    data = _try_load(
        Path(f"results/phase16/general_qa/phase16_qa_seed{seed}_summary.json")
    )
    if data is not None:
        qa_summaries[seed] = data

if qa_summaries:
    fig1_path = OUT_DIR / "convergence_general_qa.png"
    _plot_convergence(
        qa_summaries,
        "Phase 16 Multi-Seed Convergence — General-QA",
        fig1_path,
    )
    print(f"  Saved: {fig1_path} ({fig1_path.stat().st_size:,} bytes)")
else:
    print("  Skipped — no General-QA convergence data found")

# ── Figure 2 — Multi-seed convergence: Legal ───────────────────────────────
print("Figure 2 — Multi-seed convergence: Legal")

legal_summaries: dict[int, list[dict]] = {}
for seed in SEEDS:
    data = _try_load(
        Path(f"results/phase16/legal/phase16_legal_seed{seed}_summary.json")
    )
    if data is not None:
        legal_summaries[seed] = data

if legal_summaries:
    fig2_path = OUT_DIR / "convergence_legal.png"
    _plot_convergence(
        legal_summaries,
        "Phase 16 Multi-Seed Convergence — Legal",
        fig2_path,
    )
    print(f"  Saved: {fig2_path} ({fig2_path.stat().st_size:,} bytes)")
else:
    print("  Skipped — no Legal convergence data found")

# ── Figure 3 & 4 — System comparison bar charts ────────────────────────────
print("Figure 3 & 4 — System comparison bar charts (held-out)")

held_out_data = _try_load(Path("results/phase16/held_out_phase16.json"))

# Colour palette: baseline grey, v1 blue, v2 orange/green
_SYSTEM_COLOURS: dict[str, str] = {
    "Direct LLM": "#9E9E9E",
    "Naive RAG": "#BDBDBD",
    "Conservative RAG": "#78909C",
    "Iterative RAG": "#956cb4",
    "GA v1 best": "#4878d0",
    "GA v2 consensus": "#ee854a",
    "GA v2 best": "#6acc65",
}
_DEFAULT_COLOUR = "#d65f5f"


def _get_colour(name: str) -> str:
    """Return colour for a system name, falling back to default."""
    for key, colour in _SYSTEM_COLOURS.items():
        if key.lower() in name.lower():
            return colour
    return _DEFAULT_COLOUR


def _plot_comparison(
    systems: list[dict], title: str, out_path: Path
) -> None:
    """Bar chart: systems on x-axis, fitness on y-axis with F1 annotations."""
    names = [s["name"] for s in systems]
    fitness_vals = [s.get("fitness", 0.0) for s in systems]
    f1_vals = [s.get("f1", None) for s in systems]
    colours = [_get_colour(n) for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, fitness_vals, color=colours, alpha=0.85, width=0.6)

    for i, (bar, f1) in enumerate(zip(bars, f1_vals)):
        # Annotate fitness value
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{fitness_vals[i]:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
        # Annotate F1 if available
        if f1 is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"F1={f1:.3f}",
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Fitness", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if held_out_data is not None:
    # Expect held_out_data to have "general_qa" and "legal" keys,
    # each containing a list of system dicts with "name", "fitness", "f1", etc.
    qa_systems = held_out_data.get("general_qa")
    legal_systems = held_out_data.get("legal")

    if qa_systems:
        fig3_path = OUT_DIR / "comparison_general_qa.png"
        _plot_comparison(
            qa_systems,
            "Phase 16 System Comparison — General-QA (Held-Out)",
            fig3_path,
        )
        print(f"  Saved: {fig3_path} ({fig3_path.stat().st_size:,} bytes)")
    else:
        print("  Skipped Figure 3 — no 'general_qa' key in held-out data")

    if legal_systems:
        fig4_path = OUT_DIR / "comparison_legal.png"
        _plot_comparison(
            legal_systems,
            "Phase 16 System Comparison — Legal (Held-Out)",
            fig4_path,
        )
        print(f"  Saved: {fig4_path} ({fig4_path.stat().st_size:,} bytes)")
    else:
        print("  Skipped Figure 4 — no 'legal' key in held-out data")
else:
    print("  Skipped Figures 3 & 4 — held-out data not found")

# ── Figure 5 — Extended metrics (BERTScore + RAGAS Faithfulness) ────────────
print("Figure 5 — Extended metrics (BERTScore + RAGAS Faithfulness)")

ragas_data = _try_load(Path("results/phase16/ragas_faithfulness.json"))
stats_data = _try_load(Path("results/phase16/statistical_analysis.json"))

# Build extended metrics from available sources
extended_systems: list[dict] = []

if ragas_data is not None:
    # Expect ragas_data to be a list of dicts or a dict with "systems" key
    if isinstance(ragas_data, list):
        extended_systems = ragas_data
    elif isinstance(ragas_data, dict) and "systems" in ragas_data:
        extended_systems = ragas_data["systems"]
    elif isinstance(ragas_data, dict):
        # Try to reconstruct from top-level keys
        for name, metrics in ragas_data.items():
            if isinstance(metrics, dict):
                extended_systems.append({"name": name, **metrics})

if extended_systems:
    names = [s["name"] for s in extended_systems]
    bertscore_vals = [s.get("bertscore_f1", 0.0) for s in extended_systems]
    faithfulness_vals = [
        s.get("ragas_faithfulness", 0.0) for s in extended_systems
    ]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_b = ax.bar(
        x - width / 2, bertscore_vals, width,
        label="BERTScore F1", color="#4878d0", alpha=0.85,
    )
    bars_f = ax.bar(
        x + width / 2, faithfulness_vals, width,
        label="RAGAS Faithfulness", color="#ee854a", alpha=0.85,
    )

    # Annotate values
    for bars in [bars_b, bars_f]:
        for bar in bars:
            val = bar.get_height()
            if abs(val) > 0.001:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Extended Metrics — BERTScore & RAGAS Faithfulness", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    fig5_path = OUT_DIR / "extended_metrics.png"
    fig.savefig(fig5_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig5_path} ({fig5_path.stat().st_size:,} bytes)")
else:
    print("  Skipped — no extended metrics data found")

# ── Final inventory ─────────────────────────────────────────────────────────
print(f"\n{'=' * 55}")
print(f"Phase 16 figures directory: {OUT_DIR}")
saved_files = sorted(OUT_DIR.iterdir()) if OUT_DIR.exists() else []
if saved_files:
    for f in saved_files:
        if f.is_file():
            print(f"  {f.name:40s} {f.stat().st_size:>8,} bytes")
else:
    print("  (no figures generated — result data not yet available)")
print(f"{'=' * 55}")

sys.exit(0)
