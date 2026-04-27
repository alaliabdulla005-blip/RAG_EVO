"""Generate Phase 16 dissertation figures.

Produces eight publication-quality PNGs at 300 DPI under
``results/phase16/figures/``:

  1. phase16_qa_convergence.png              — 3-panel QA convergence, one per seed
  2. phase16_qa_held_out_comparison.png      — grouped QA held-out bar chart
  3. phase16_legal_convergence.png           — Legal seed 42 convergence
  4. phase16_cross_seed_basin.png            — gen-99 chromosome scatter
  5. phase16_extended_metrics.png            — Faithfulness + BERTScore F1
  6. phase12_qa_convergence.png              — Phase 12 QA single-seed curve
  7. phase16_legal_held_out_comparison.png   — Phase 16 Legal held-out bars
  8. phase16_gene_heatmap.png                — gene heatmap with Phase 16 cols

DO NOT run automatically — run after all evaluations are complete.

Run from project root:
    python scripts/generate_phase16_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT_DIR = Path("results/phase16/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

DPI = 300
SEEDS_QA = [42, 123, 7]

# Reference values (external — dissertation context, not recomputed here).
PHASE12_QA_BEST = 0.240
PHASE12_QA_HELD_OUT = 0.150
NAIVE_RAG_QA_HELD_OUT = 0.155
PHASE13_LEGAL_BEST = 0.185
NAIVE_RAG_QA_PHASE12 = 0.1181

# Palette.
GA_BLUE = "#2196F3"
XF_ORANGE = "#FF9800"
BASELINE_GREY = "#9E9E9E"
SEED_COLOURS = {42: "#E53935", 123: "#1E88E5", 7: "#43A047"}
FAITH_CORAL = "#FF7043"
BERT_TEAL = "#00897B"


def _try_load(path: Path):
    if not path.exists():
        print(f"  WARNING: {path} missing — skipping dependent figure(s)")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — QA convergence, one panel per seed
# ──────────────────────────────────────────────────────────────────────────────
def fig1_qa_convergence() -> None:
    print("Figure 1 — QA convergence (3 panels)")
    summaries: dict[int, list[dict]] = {}
    for seed in SEEDS_QA:
        p = Path(
            f"results/phase16/general_qa/phase16_qa_seed{seed}_summary.json"
        )
        data = _try_load(p)
        if data is None:
            return
        summaries[seed] = data

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, seed in zip(axes, SEEDS_QA):
        recs = summaries[seed]
        gens = [r["generation_idx"] for r in recs]
        best = [r["best_fitness"] for r in recs]
        mean = [r["mean_fitness"] for r in recs]
        colour = GA_BLUE
        ax.plot(gens, best, color=colour, linewidth=2, label="Best")
        ax.plot(
            gens, mean,
            color=colour, linewidth=1.4, linestyle="--", alpha=0.55,
            label="Mean",
        )
        ax.axhline(
            PHASE12_QA_BEST,
            color="#555555", linewidth=1, linestyle=":",
            label=f"Phase 12 best ({PHASE12_QA_BEST:.3f})",
        )
        ax.set_title(f"Seed {seed}", fontsize=12)
        ax.set_xlim(0, 99)
        ax.set_ylim(0.15, 0.35)
        ax.set_xlabel("Generation")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Fitness")
    axes[0].legend(loc="lower right", fontsize=9, frameon=False)
    _save(fig, OUT_DIR / "phase16_qa_convergence.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — QA held-out bar chart (9 systems, grouped by colour)
# ──────────────────────────────────────────────────────────────────────────────
def fig2_qa_held_out() -> None:
    print("Figure 2 — QA held-out comparison (9 systems)")
    d = _try_load(Path("results/phase16/held_out_phase16.json"))
    if d is None:
        return

    group_map = {
        "GA-P16-QA-S42": GA_BLUE,
        "GA-P16-QA-S123": GA_BLUE,
        "GA-P16-QA-S7": GA_BLUE,
        "P12-Best": XF_ORANGE,
        "P13-Best-on-QA": XF_ORANGE,
        "GA-P16-Legal-S42-on-QA": XF_ORANGE,
        "Naive-RAG": BASELINE_GREY,
        "Conservative-RAG": BASELINE_GREY,
        "Iterative-RAG": BASELINE_GREY,
    }
    systems = [
        r for r in d["results"]["general_qa"]
        if r["name"] != "Direct-LLM"
    ]
    systems.sort(key=lambda r: r["mean_fitness"], reverse=True)

    names = [r["name"] for r in systems]
    fitness = [r["mean_fitness"] for r in systems]
    colours = [group_map[n] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(names))
    bars = ax.barh(y, fitness, color=colours, edgecolor="white", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()

    for bar, f in zip(bars, fitness):
        ax.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{f:.3f}",
            va="center", fontsize=9, color="#333333",
        )

    ax.axvline(
        PHASE12_QA_HELD_OUT,
        linestyle="--", color="#555555", linewidth=1,
    )

    ax.set_xlabel("Mean held-out fitness (n=300)")
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.3)

    ax.legend(
        handles=[
            Patch(color=GA_BLUE, label="GA-P16 native"),
            Patch(color=XF_ORANGE, label="Cross-phase / transfer"),
            Patch(color=BASELINE_GREY, label="Manual baselines"),
            Line2D(
                [], [], color="#555555", linestyle="--",
                label=f"Phase 12 held-out ({PHASE12_QA_HELD_OUT:.3f})",
            ),
        ],
        loc="lower right", frameon=False, fontsize=9,
    )
    _save(fig, OUT_DIR / "phase16_qa_held_out_comparison.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Legal seed 42 convergence (single panel)
# ──────────────────────────────────────────────────────────────────────────────
def fig3_legal_convergence() -> None:
    print("Figure 3 — Legal seed 42 convergence")
    d = _try_load(
        Path("results/phase16/legal/phase16_legal_seed42_summary.json")
    )
    if d is None:
        return

    gens = [r["generation_idx"] for r in d]
    best = [r["best_fitness"] for r in d]
    mean = [r["mean_fitness"] for r in d]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gens, best, color=GA_BLUE, linewidth=2, label="Best")
    ax.plot(
        gens, mean,
        color=GA_BLUE, linewidth=1.4, linestyle="--", alpha=0.55,
        label="Mean",
    )
    ax.axhline(
        PHASE13_LEGAL_BEST,
        color="#555555", linewidth=1, linestyle=":",
        label=f"Phase 13 best ({PHASE13_LEGAL_BEST:.3f})",
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_xlim(0, 99)
    ax.set_ylim(0.0, 0.30)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    _save(fig, OUT_DIR / "phase16_legal_convergence.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — cross-seed basin scatter (gen-99 individuals, chunk_size × overlap)
# ──────────────────────────────────────────────────────────────────────────────
def fig4_cross_seed_basin() -> None:
    print("Figure 4 — cross-seed gen-99 basin scatter")
    pops: dict[int, list[dict]] = {}
    for seed in SEEDS_QA:
        p = Path(
            f"results/phase16/general_qa/seed_{seed}/gen_099/population.json"
        )
        data = _try_load(p)
        if data is None:
            return
        pops[seed] = data

    all_fit = np.array([ind["fitness"] for pop in pops.values() for ind in pop])
    fit_min, fit_max = float(all_fit.min()), float(all_fit.max())
    fit_span = max(fit_max - fit_min, 1e-9)

    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for seed in SEEDS_QA:
        pop = pops[seed]
        chunks = np.array([ind["config"]["chunk_size"] for ind in pop])
        overlaps = np.array([ind["config"]["overlap"] for ind in pop])
        fits = np.array([ind["fitness"] for ind in pop])
        jitter = rng.uniform(-5.0, 5.0, size=len(pop))
        sizes = 30 + (fits - fit_min) / fit_span * 220
        ax.scatter(
            chunks + jitter, overlaps,
            s=sizes, c=SEED_COLOURS[seed],
            alpha=0.65, edgecolors="white", linewidths=0.5,
            label=f"Seed {seed}",
        )

    # Basin annotations — anchor arrows on the best individual per group,
    # which robustly identifies the high-fitness mode even when the
    # population is multimodal (simple centroids land in empty space).
    def _best_xy(seeds: list[int]) -> tuple[float, float]:
        best = max(
            (ind for s in seeds for ind in pops[s]),
            key=lambda ind: ind["fitness"],
        )
        return (
            float(best["config"]["chunk_size"]),
            float(best["config"]["overlap"]),
        )

    b42 = _best_xy([42])
    b_123_7 = _best_xy([123, 7])
    ax.annotate(
        "Seed 42 basin",
        xy=b42, xytext=(b42[0] + 40, b42[1] - 0.12),
        fontsize=11, color=SEED_COLOURS[42],
        arrowprops={"arrowstyle": "->", "color": SEED_COLOURS[42], "lw": 1.1},
    )
    ax.annotate(
        "Seeds 123/7 basin",
        xy=b_123_7, xytext=(b_123_7[0] - 180, b_123_7[1] - 0.08),
        fontsize=11, color="#333333",
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.1},
    )

    ax.set_xticks([256, 384, 448, 512])
    ax.set_xlabel("chunk_size (with ±5 jitter)")
    ax.set_ylabel("overlap")
    ax.set_ylim(-0.02, 0.55)
    ax.grid(axis="both", alpha=0.2)
    ax.legend(loc="upper right", frameon=False, fontsize=10, title="Seed")
    _save(fig, OUT_DIR / "phase16_cross_seed_basin.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5 — extended metrics (RAGAS Faithfulness + BERTScore F1, 8 systems)
# ──────────────────────────────────────────────────────────────────────────────
def fig5_extended_metrics() -> None:
    print("Figure 5 — extended metrics (Faithfulness + BERTScore)")
    ragas = _try_load(Path("results/phase16/ragas_faithfulness.json"))
    held = _try_load(Path("results/phase16/held_out_phase16.json"))
    if ragas is None or held is None:
        return

    bs_lookup: dict[tuple[str, str], float] = {}
    for track in ("general_qa", "legal"):
        for row in held["results"][track]:
            bs_lookup[(track, row["name"])] = row.get(
                "mean_bertscore_f1", float("nan")
            )

    order: list[tuple[str, str, str]] = []  # (track, name, display_label)
    for s in ragas["general_qa"]["systems"]:
        order.append(("general_qa", s["name"], s["name"]))
    for s in ragas["legal"]["systems"]:
        order.append(("legal", s["name"], s["name"]))

    faiths = [
        next(
            x for x in ragas[track]["systems"] if x["name"] == name
        )["mean_faithfulness"]
        for (track, name, _) in order
    ]
    berts = [bs_lookup[(track, name)] for (track, name, _) in order]

    n = len(order)
    x = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        x - width / 2, faiths,
        width, color=FAITH_CORAL, label="RAGAS Faithfulness",
    )
    ax.bar(
        x + width / 2, berts,
        width, color=BERT_TEAL, label="BERTScore F1",
    )

    # Track separation.
    qa_n = sum(1 for t, _, _ in order if t == "general_qa")
    split = qa_n - 0.5
    ax.axvline(split, color="#333333", linewidth=0.8, linestyle="-")
    ax.axvspan(
        split, n - 0.5, color="black", alpha=0.04,
    )

    # Track labels at the top of each half.
    y_label = 0.97
    ax.text(
        (qa_n - 1) / 2, y_label, "General-QA",
        ha="center", va="top", transform=ax.get_xaxis_transform(),
        fontsize=11, fontweight="bold", color="#555555",
    )
    ax.text(
        qa_n + (n - qa_n - 1) / 2, y_label, "Legal",
        ha="center", va="top", transform=ax.get_xaxis_transform(),
        fontsize=11, fontweight="bold", color="#555555",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, label in order], rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    _save(fig, OUT_DIR / "phase16_extended_metrics.png")


def fig6_phase12_qa_convergence() -> None:
    """Phase 12 General-QA GA run (15 gens, pop 20) — best + mean + naive ref."""
    print("Figure 6 — Phase 12 QA convergence")
    d = _try_load(
        Path("results/phase12/general_qa/phase12_summary.json")
    )
    if d is None:
        return

    gens = [r["generation_idx"] for r in d]
    best = [r["best_fitness"] for r in d]
    mean = [r["mean_fitness"] for r in d]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gens, best, color=GA_BLUE, linewidth=2, label="Best")
    ax.plot(
        gens, mean,
        color=GA_BLUE, linewidth=1.4, linestyle="--", alpha=0.55,
        label="Mean",
    )
    ax.axhline(
        NAIVE_RAG_QA_PHASE12,
        color="#555555", linewidth=1, linestyle=":",
        label=f"Naive-RAG baseline ({NAIVE_RAG_QA_PHASE12:.3f})",
    )
    # Annotate global best achieved at gen 6.
    best_val = max(best)
    best_gen = gens[best.index(best_val)]
    ax.annotate(
        f"Global best = {best_val:.4f}",
        xy=(best_gen, best_val),
        xytext=(best_gen + 2.5, best_val - 0.018),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.0},
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_xlim(0, max(gens))
    ax.set_ylim(0.08, 0.26)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    _save(fig, OUT_DIR / "phase12_qa_convergence.png")


def fig7_phase16_legal_held_out() -> None:
    """Phase 16 Legal held-out (n=300) — 10 systems, 3-colour groups."""
    print("Figure 7 — Phase 16 Legal held-out comparison")
    d = _try_load(Path("results/phase16/held_out_phase16.json"))
    if d is None:
        return

    group_map = {
        "GA-P16-Legal-S42": GA_BLUE,
        "P13-Best": XF_ORANGE,
        "P12-Best-on-Legal": XF_ORANGE,
        "GA-P16-QA-S42-on-Legal": XF_ORANGE,
        "GA-P16-QA-S123-on-Legal": XF_ORANGE,
        "GA-P16-QA-S7-on-Legal": XF_ORANGE,
        "Direct-LLM": BASELINE_GREY,
        "Naive-RAG": BASELINE_GREY,
        "Conservative-RAG": BASELINE_GREY,
        "Iterative-RAG": BASELINE_GREY,
    }
    systems = list(d["results"]["legal"])
    systems.sort(key=lambda r: r["mean_fitness"], reverse=True)

    names = [r["name"] for r in systems]
    fitness = [r["mean_fitness"] for r in systems]
    colours = [group_map[n] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5.8))
    y = np.arange(len(names))
    bars = ax.barh(y, fitness, color=colours, edgecolor="white", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()

    # Annotate each bar's numerical fitness just outside its tip.
    xmax = max(fitness)
    for bar, f in zip(bars, fitness):
        x_text = bar.get_width() + (0.003 if f >= 0 else -0.003)
        ha = "left" if f >= 0 else "right"
        ax.text(
            x_text,
            bar.get_y() + bar.get_height() / 2,
            f"{f:+.3f}",
            va="center", ha=ha, fontsize=9, color="#333333",
        )

    # Zero reference — useful because several systems have negative fitness.
    ax.axvline(0, color="#555555", linewidth=0.8, linestyle="-")

    ax.set_xlabel("Mean held-out fitness (n=300)")
    ax.set_xlim(min(fitness) - 0.02, xmax + 0.03)
    ax.grid(axis="x", alpha=0.3)

    ax.legend(
        handles=[
            Patch(color=GA_BLUE, label="GA-P16 native (Legal)"),
            Patch(color=XF_ORANGE, label="Cross-phase / transfer"),
            Patch(color=BASELINE_GREY, label="Manual baselines"),
        ],
        loc="lower right", frameon=False, fontsize=9,
    )
    _save(fig, OUT_DIR / "phase16_legal_held_out_comparison.png")


def fig8_phase16_gene_heatmap() -> None:
    """Domain-universal vs domain-specific genes across Phases 12, 13, 16."""
    print("Figure 8 — Gene heatmap with Phase 16 columns")

    # Raw values per column. Phase 12/13 values copied from the existing
    # generate_dissertation_figures.py (the authoritative source); Phase 16
    # values are the best chromosomes from the seed-42 runs.
    columns = {
        "General-QA\n(Phase 12)": {
            "chunk_size": 512, "overlap": 0.0, "top_k": 19,
            "prompt_strategy": "chain-of-thought", "temperature": 0.6743,
        },
        "Legal\n(Phase 13)": {
            "chunk_size": 384, "overlap": 0.2, "top_k": 20,
            "prompt_strategy": "chain-of-thought", "temperature": 1.0,
        },
        "General-QA\n(Phase 16, S42)": {
            "chunk_size": 256, "overlap": 0.382, "top_k": 20,
            "prompt_strategy": "legal-citation", "temperature": 0.814,
        },
        "Legal\n(Phase 16, S42)": {
            "chunk_size": 384, "overlap": 0.133, "top_k": 20,
            "prompt_strategy": "legal-citation", "temperature": 1.0,
        },
    }
    gene_labels = [
        "chunk_size", "overlap", "top_k", "prompt_strategy", "temperature",
    ]

    # Display strings for cell text.
    def _fmt(gene: str, v) -> str:
        if gene == "chunk_size":
            return str(int(v))
        if gene == "overlap":
            return f"{v:.3f}" if v not in (0.0, 0.2) else f"{v:.1f}"
        if gene == "top_k":
            return str(int(v))
        if gene == "prompt_strategy":
            return {"chain-of-thought": "CoT",
                    "legal-citation": "legal-cit",
                    "zero-shot": "zero-shot"}.get(v, str(v))
        if gene == "temperature":
            return f"{v:.3f}" if v != 1.0 else "1.0"
        return str(v)

    # Normalisation — widened to cover Phase 16 v2 chromosome ranges.
    # prompt_strategy now has three levels (was two in earlier phases).
    def _norm(gene: str, v) -> float:
        if gene == "chunk_size":
            return (v - 256) / (512 - 256)
        if gene == "overlap":
            return v / 0.5  # Phase 16 v2 upper bound
        if gene == "top_k":
            return (v - 1) / (20 - 1)
        if gene == "prompt_strategy":
            return {"zero-shot": 0.0,
                    "chain-of-thought": 0.5,
                    "legal-citation": 1.0}.get(v, 0.5)
        if gene == "temperature":
            return float(v)
        return float(v)

    col_names = list(columns.keys())
    norm_values = np.array([
        [_norm(g, columns[c][g]) for c in col_names] for g in gene_labels
    ])

    fig, ax = plt.subplots(figsize=(9, 5.2))
    im = ax.imshow(norm_values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, fontsize=10)
    ax.set_yticks(range(len(gene_labels)))
    ax.set_yticklabels(gene_labels, fontsize=10)

    for i, gene in enumerate(gene_labels):
        for j, col in enumerate(col_names):
            raw = columns[col][gene]
            ax.text(
                j, i, _fmt(gene, raw),
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="black" if norm_values[i, j] < 0.65 else "white",
            )

    plt.colorbar(
        im, ax=ax, label="Normalised value", fraction=0.035, pad=0.04
    )
    ax.set_title("Domain-Universal vs Domain-Specific Genes (Phases 12, 13, 16)")
    _save(fig, OUT_DIR / "phase16_gene_heatmap.png")


def main() -> None:
    print(f"Generating Phase 16 figures → {OUT_DIR}/")
    fig1_qa_convergence()
    fig2_qa_held_out()
    fig3_legal_convergence()
    fig4_cross_seed_basin()
    fig5_extended_metrics()
    fig6_phase12_qa_convergence()
    fig7_phase16_legal_held_out()
    fig8_phase16_gene_heatmap()
    print("Done.")


if __name__ == "__main__":
    main()
