"""Convergence plot from Phase 13 Legal GA data.

Reads results/phase13/legal_qa/phase13_summary.json and saves
results/figures/convergence_plot_legal.png.

Run from project root:
    python scripts/plot_convergence_legal.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

SUMMARY_PATH = Path("results/phase13/legal_qa/phase13_summary.json")
OUT_DIR = Path("results/figures")
OUT_PATH = OUT_DIR / "convergence_plot_legal.png"

records = json.loads(SUMMARY_PATH.read_text())

generations = [r["generation_idx"] + 1 for r in records]  # 1-indexed
best = [r["best_fitness"] for r in records]
mean = [r["mean_fitness"] for r in records]
worst = [r["worst_fitness"] for r in records]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(generations, best, color="blue", linewidth=2, label="Best fitness")
ax.plot(generations, mean, color="orange", linewidth=1.5,
        linestyle="--", label="Mean fitness")
ax.plot(generations, worst, color="grey", linewidth=1,
        linestyle=":", label="Worst fitness")

# Horizontal iterative RAG baseline reference line
ax.axhline(y=0.066, color="red", linestyle="--", linewidth=1.2,
           label="Iterative RAG baseline (0.066)")

# Vertical line at generation 4 (0-indexed gen 3 = 1-indexed gen 4)
ax.axvline(x=4, color="green", linestyle="--", linewidth=1.2,
           label="Global best (gen 4)")

ax.set_xlim(1, 15)
ax.set_ylim(-0.05, 0.25)
ax.set_xlabel("Generation")
ax.set_ylabel("Fitness")
ax.set_title("GA Legal Track Convergence — 15 Generations, Population 20")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
