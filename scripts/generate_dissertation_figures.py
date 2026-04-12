"""Generate all dissertation figures from saved result JSON files.

No API calls. Reads existing result files and generates 7 publication-quality
matplotlib figures. Missing files are skipped with a warning.

Run from project root:
    python scripts/generate_dissertation_figures.py

Outputs (all in results/figures/):
    convergence_plot.png          (Figure 1, already exists)
    convergence_plot_legal.png    (Figure 2, already exists)
    general_qa_comparison.png     (Figure 3)
    legal_comparison.png          (Figure 4)
    cross_track_radar.png         (Figure 5)
    gene_heatmap.png              (Figure 6)
    fitness_improvement.png       (Figure 7)
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 150


def _try_load(path: Path) -> dict | list | None:
    """Load a JSON file, returning None with a warning if missing."""
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping dependent figure(s)")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ── Figure 1 — GA Phase 12 Convergence ───────────────────────────────────────
print("Figure 1 — GA Phase 12 Convergence")
fig1_path = OUT_DIR / "convergence_plot.png"
if fig1_path.exists():
    print(f"  Already exists: {fig1_path} ({fig1_path.stat().st_size:,} bytes)")
else:
    data = _try_load(Path("results/phase12/general_qa/phase12_summary.json"))
    if data is not None:
        generations = [r["generation_idx"] + 1 for r in data]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(generations, [r["best_fitness"] for r in data],
                color="blue", linewidth=2, label="Best fitness")
        ax.plot(generations, [r["mean_fitness"] for r in data],
                color="orange", linewidth=1.5, linestyle="--", label="Mean fitness")
        ax.plot(generations, [r["worst_fitness"] for r in data],
                color="grey", linewidth=1, linestyle=":", label="Worst fitness")
        best_gen = max(data, key=lambda r: r["best_fitness"])["generation_idx"] + 1
        ax.axvline(x=best_gen, color="green", linestyle="--", linewidth=1.2,
                   label=f"Global best (gen {best_gen})")
        ax.set_xlim(1, 15)
        ax.set_ylim(0.0, 0.30)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("GA Phase 12 Convergence — 15 Generations, Population 20")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.savefig(fig1_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fig1_path}")

# ── Figure 2 — GA Phase 13 Legal Convergence ─────────────────────────────────
print("Figure 2 — GA Phase 13 Legal Convergence")
fig2_path = OUT_DIR / "convergence_plot_legal.png"
if fig2_path.exists():
    print(f"  Already exists: {fig2_path} ({fig2_path.stat().st_size:,} bytes)")
else:
    data = _try_load(Path("results/phase13/legal_qa/phase13_summary.json"))
    if data is not None:
        generations = [r["generation_idx"] + 1 for r in data]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(generations, [r["best_fitness"] for r in data],
                color="blue", linewidth=2, label="Best fitness")
        ax.plot(generations, [r["mean_fitness"] for r in data],
                color="orange", linewidth=1.5, linestyle="--", label="Mean fitness")
        ax.plot(generations, [r["worst_fitness"] for r in data],
                color="grey", linewidth=1, linestyle=":", label="Worst fitness")
        ax.axhline(y=0.066, color="red", linestyle="--", linewidth=1.2,
                   label="Iterative RAG baseline (0.066)")
        ax.axvline(x=4, color="green", linestyle="--", linewidth=1.2,
                   label="Global best (gen 4)")
        ax.set_xlim(1, 15)
        ax.set_ylim(-0.05, 0.25)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("GA Legal Track Convergence — 15 Generations, Population 20")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.savefig(fig2_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fig2_path}")

# ── Figure 3 — General-QA comparison bar chart ───────────────────────────────
print("Figure 3 — General-QA comparison bar chart")

_NAME_MAP = {
    "Strong manual RAG": "Conservative RAG",
    "Conservative manual RAG": "Conservative RAG",
}


def _display_name(name: str) -> str:
    return _NAME_MAP.get(name, name)


baseline_data = _try_load(Path("results/ablations/baseline_comparison.json"))
iter_rag_data = _try_load(Path("results/ablations/iterative_rag.json"))

if baseline_data is not None and iter_rag_data is not None:
    systems_gen = [
        {
            "name": _display_name(s["name"]),
            "fitness": s["fitness"],
            "cit_f1": s["metrics"]["mean_citation_f1"],
            "ret_r": s["metrics"]["mean_retrieval_recall_at_k"],
        }
        for s in baseline_data["systems"]
    ]
    systems_gen.insert(
        3,
        {
            "name": "Iterative RAG",
            "fitness": iter_rag_data["fitness"],
            "cit_f1": iter_rag_data["metrics"]["mean_citation_f1"],
            "ret_r": iter_rag_data["metrics"]["mean_retrieval_recall_at_k"],
        },
    )

    fig3_path = OUT_DIR / "general_qa_comparison.png"
    metric_labels = ["Fitness", "Cit-F1", "Ret-R@k"]
    metric_keys = ["fitness", "cit_f1", "ret_r"]
    colours = ["#4878d0", "#ee854a", "#6acc65", "#d65f5f", "#956cb4"]
    sys_names = [s["name"] for s in systems_gen]

    x = np.arange(len(metric_labels))
    width = 0.15
    n = len(systems_gen)
    offsets = (np.arange(n) - (n - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (s, colour) in enumerate(zip(systems_gen, colours)):
        vals = [s[k] for k in metric_keys]
        bars = ax.bar(x + offsets[i], vals, width, label=s["name"],
                      color=colour, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val != 0.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("General-QA Track — System Comparison")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(fig3_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig3_path} ({fig3_path.stat().st_size:,} bytes)")

# ── Figure 4 — Legal comparison bar chart ────────────────────────────────────
print("Figure 4 — Legal comparison bar chart")

legal_baseline_data = _try_load(Path("results/phase14/legal_baselines.json"))
iter_legal_data = _try_load(Path("results/phase13/iterative_rag_legal.json"))

if legal_baseline_data is not None and iter_legal_data is not None:
    systems_leg = [
        {
            "name": _display_name(s["name"]),
            "fitness": s["fitness"],
            "cit_f1": s["metrics"]["mean_citation_f1"],
            "ret_r": s["metrics"]["mean_retrieval_recall_at_k"],
        }
        for s in legal_baseline_data["systems"]
    ]
    systems_leg.insert(
        3,
        {
            "name": "Iterative RAG",
            "fitness": iter_legal_data["fitness"],
            "cit_f1": iter_legal_data["metrics"]["mean_citation_f1"],
            "ret_r": iter_legal_data["metrics"]["mean_retrieval_recall_at_k"],
        },
    )

    fig4_path = OUT_DIR / "legal_comparison.png"
    x = np.arange(len(metric_labels))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (s, colour) in enumerate(zip(systems_leg, colours)):
        vals = [s[k] for k in metric_keys]
        bars = ax.bar(x + offsets[i], vals, width, label=s["name"],
                      color=colour, alpha=0.85)
        for bar, val in zip(bars, vals):
            if abs(val) > 0.001:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("Legal Track — System Comparison")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(fig4_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig4_path} ({fig4_path.stat().st_size:,} bytes)")

# ── Figure 5 — Cross-track radar chart ───────────────────────────────────────
print("Figure 5 — Cross-track radar chart")

summary_data = _try_load(Path("results/phase14/cross_track_summary.json"))

if summary_data is not None:
    rows_by_name: dict[str, dict] = {r["system"]: r for r in summary_data["rows"]}

    # Efficiency fallback for rows with null input_tokens
    def _efficiency(row: dict, fallback_name: str) -> float:
        toks = row.get("input_tokens")
        if toks is None:
            fb = rows_by_name.get(fallback_name, {})
            toks = fb.get("input_tokens") or 0.0
        return max(0.0, min(1.0, 1.0 - toks / 10000.0))

    radar_series = [
        {
            "label": "GA Phase 12",
            "row": rows_by_name.get("GA Phase 12 best", {}),
            "efficiency_fallback": "GA-optimised",
            "colour": "#4878d0",
        },
        {
            "label": "GA Phase 13",
            "row": rows_by_name.get("GA Phase 13 best", {}),
            "efficiency_fallback": "GA-optimised legal",
            "colour": "#ee854a",
        },
        {
            "label": "Ph12 → Legal",
            "row": rows_by_name.get(
                "Phase 12 General-QA GA best → legal", {}
            ),
            "efficiency_fallback": None,
            "colour": "#6acc65",
        },
        {
            "label": "Ph13 → General",
            "row": rows_by_name.get(
                "Phase 13 Legal GA best → general", {}
            ),
            "efficiency_fallback": None,
            "colour": "#d65f5f",
        },
    ]

    axes_labels = ["Fitness", "F1", "Cit-F1", "Ret-R@5", "Efficiency"]
    n_axes = len(axes_labels)
    angles = [2 * math.pi * i / n_axes for i in range(n_axes)]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for series in radar_series:
        row = series["row"]
        if not row:
            continue
        fitness = row.get("fitness") or 0.0
        f1 = row.get("f1") or 0.0
        cit_f1 = row.get("cit_f1") or 0.0
        ret_r = row.get("ret_r_at_k") or 0.0
        eff = _efficiency(row, series["efficiency_fallback"] or "")
        vals = [fitness, f1, cit_f1, ret_r, eff]
        vals_closed = vals + [vals[0]]
        ax.plot(angles_closed, vals_closed, linewidth=2,
                color=series["colour"], label=series["label"])
        ax.fill(angles_closed, vals_closed, alpha=0.1, color=series["colour"])

    ax.set_xticks(angles)
    ax.set_xticklabels(axes_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title(
        "Cross-Track Configuration Transfer", pad=20, fontsize=12
    )
    ax.grid(True, alpha=0.4)

    fig5_path = OUT_DIR / "cross_track_radar.png"
    fig.savefig(fig5_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig5_path} ({fig5_path.stat().st_size:,} bytes)")

# ── Figure 6 — Gene heatmap ───────────────────────────────────────────────────
print("Figure 6 — Gene heatmap")

gene_configs = {
    "General-QA\n(Phase 12)": {
        "chunk_size": 512,
        "overlap": 0.0,
        "top_k": 19,
        "prompt_strategy": "chain-of-thought",
        "temperature": 0.6743266807244299,
    },
    "Legal\n(Phase 13)": {
        "chunk_size": 384,
        "overlap": 0.2,
        "top_k": 20,
        "prompt_strategy": "chain-of-thought",
        "temperature": 1.0,
    },
}

gene_labels = ["chunk_size", "overlap", "top_k", "prompt_strategy", "temperature"]
gene_actual = {
    "General-QA\n(Phase 12)": {
        "chunk_size": "512",
        "overlap": "0.0",
        "top_k": "19",
        "prompt_strategy": "CoT",
        "temperature": "0.674",
    },
    "Legal\n(Phase 13)": {
        "chunk_size": "384",
        "overlap": "0.2",
        "top_k": "20",
        "prompt_strategy": "CoT",
        "temperature": "1.0",
    },
}


def _norm_gene(gene: str, val: float) -> float:
    if gene == "chunk_size":
        return (val - 256) / (512 - 256)
    if gene == "overlap":
        return val / 0.3
    if gene == "top_k":
        return (val - 1) / (20 - 1)
    if gene == "prompt_strategy":
        return 0.0 if val == "zero-shot" else 1.0
    if gene == "temperature":
        return float(val)
    return float(val)


col_names = list(gene_configs.keys())
norm_values = np.array([
    [_norm_gene(g, gene_configs[col][g]) for col in col_names]
    for g in gene_labels
])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(norm_values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

ax.set_xticks(range(len(col_names)))
ax.set_xticklabels(col_names, fontsize=10)
ax.set_yticks(range(len(gene_labels)))
ax.set_yticklabels(gene_labels, fontsize=10)

for i, gene in enumerate(gene_labels):
    for j, col in enumerate(col_names):
        ax.text(
            j, i, gene_actual[col][gene],
            ha="center", va="center",
            fontsize=11, fontweight="bold",
            color="black" if norm_values[i, j] < 0.7 else "white",
        )

plt.colorbar(im, ax=ax, label="Normalised value", fraction=0.046, pad=0.04)
ax.set_title("Domain-Universal vs Domain-Specific Genes")
fig.tight_layout()

fig6_path = OUT_DIR / "gene_heatmap.png"
fig.savefig(fig6_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {fig6_path} ({fig6_path.stat().st_size:,} bytes)")

# ── Figure 7 — Fitness delta bar chart ───────────────────────────────────────
print("Figure 7 — Fitness delta bar chart")

if summary_data is not None:
    naive_gen = next(
        (r["fitness"] for r in summary_data["rows"]
         if r["system"] == "Naive RAG" and r["track"] == "general"),
        0.107,
    )
    naive_leg = next(
        (r["fitness"] for r in summary_data["rows"]
         if r["system"] == "Naive RAG" and r["track"] == "legal"),
        0.026,
    )

    def _get_fitness(system: str, track: str) -> float | None:
        for r in summary_data["rows"]:
            if r["system"] == system and r["track"] == track:
                return r["fitness"]
        return None

    # Map display names to system names in the summary
    delta_systems = [
        ("Direct LLM",
         _get_fitness("Direct LLM", "general"),
         _get_fitness("Direct LLM", "legal")),
        ("Conservative RAG",
         _get_fitness("Strong manual RAG", "general"),
         _get_fitness("Conservative manual RAG", "legal")),
        ("Iterative RAG",
         _get_fitness("Iterative RAG", "general"),
         _get_fitness("Iterative RAG Legal", "legal")),
        ("GA-optimised",
         _get_fitness("GA-optimised", "general"),
         _get_fitness("GA-optimised legal", "legal")),
    ]

    sys_labels = [d[0] for d in delta_systems]
    delta_gen = [
        (d[1] - naive_gen) if d[1] is not None else 0.0
        for d in delta_systems
    ]
    delta_leg = [
        (d[2] - naive_leg) if d[2] is not None else 0.0
        for d in delta_systems
    ]

    y = np.arange(len(sys_labels))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_g = ax.barh(y + height / 2, delta_gen, height, label="General-QA",
                     color="#4878d0", alpha=0.85)
    bars_l = ax.barh(y - height / 2, delta_leg, height, label="Legal",
                     color="#ee854a", alpha=0.85)

    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(sys_labels)
    ax.set_xlabel("Δ Fitness vs Naive RAG")
    ax.set_title("Fitness Gain Over Naive RAG Baseline")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)

    for bar in list(bars_g) + list(bars_l):
        val = bar.get_width()
        if abs(val) > 0.001:
            x_pos = val + 0.002 if val >= 0 else val - 0.002
            ha = "left" if val >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha=ha, fontsize=9)

    fig.tight_layout()
    fig7_path = OUT_DIR / "fitness_improvement.png"
    fig.savefig(fig7_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig7_path} ({fig7_path.stat().st_size:,} bytes)")

# ── Final inventory ───────────────────────────────────────────────────────────
print(f"\n{'=' * 55}")
print("Results figures directory:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name:40s} {f.stat().st_size:>8,} bytes")
