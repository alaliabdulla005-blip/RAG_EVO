"""Generate the RAG-Evo system architecture diagram.

Three main modules (Data, Execution, Optimisation) on a horizontal flow,
plus two side layers (Result Cache, Checkpoint System). Clean, minimal
style — rounded boxes and labelled arrows. Saved at 300 DPI to
``results/figures/system_architecture.png``.

Run from project root:
    python scripts/generate_system_architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path("results/figures/system_architecture.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "DejaVu Sans"

# Palette — soft, light fills with dark strokes.
C_DATA = "#E3F2FD"
C_EXEC = "#FFF3E0"
C_OPT = "#E8F5E9"
C_CACHE = "#F3E5F5"
C_CKPT = "#FCE4EC"

EDGE = "#333333"
ARROW_BLACK = "#333333"
ARROW_GREY = "#666666"
ARROW_PURPLE = "#6A1B9A"
ARROW_MAGENTA = "#AD1457"


def _box(ax, cx, cy, w, h, title, sub, fc):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            linewidth=1.3, edgecolor=EDGE, facecolor=fc,
        )
    )
    ax.text(
        cx, cy + h / 2 - 0.28, title,
        ha="center", va="top", fontsize=12, fontweight="bold",
    )
    ax.text(
        cx, cy - 0.1, sub,
        ha="center", va="center", fontsize=9.5, color="#222222",
    )


def _arrow(ax, start, end, label, label_xy, color=ARROW_BLACK, style="->"):
    ax.add_patch(
        FancyArrowPatch(
            start, end,
            arrowstyle=f"{style},head_width=0.32,head_length=0.48",
            linewidth=1.4, color=color,
            connectionstyle="arc3,rad=0.0",
            shrinkA=0, shrinkB=0,
        )
    )
    if label:
        ax.text(
            *label_xy, label,
            ha="center", va="center", fontsize=10, color=color,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "none",
            },
        )


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    # Module sizes.
    mw, mh = 3.2, 2.0
    sw, sh = 3.0, 1.35

    # Positions.
    data = (2.2, 3.1)
    exec_ = (7.0, 3.1)
    opt = (11.8, 3.1)
    cache = (7.0, 6.3)
    ckpt = (11.8, 6.3)

    # Modules.
    _box(
        ax, *data, mw, mh,
        "Data Module",
        "Dataset loading\nBM25 index\nRecord caching",
        C_DATA,
    )
    _box(
        ax, *exec_, mw, mh,
        "Execution Module",
        "RAG pipeline factory\n→ PipelineResult:\nanswer, passages,\ntokens, latency",
        C_EXEC,
    )
    _box(
        ax, *opt, mw, mh,
        "Optimisation Module",
        "GA loop\nSelection + variation\nFitness coordination",
        C_OPT,
    )

    # Side layers.
    _box(
        ax, *cache, sw, sh,
        "Result Cache",
        "SHA-256 keyed\nshort-circuits evaluation",
        C_CACHE,
    )
    _box(
        ax, *ckpt, sw, sh,
        "Checkpoint System",
        "Population + RNG state\n(per generation)",
        C_CKPT,
    )

    # ── Main horizontal flow ────────────────────────────────────────────
    # Data → Execution
    _arrow(
        ax,
        (data[0] + mw / 2, data[1]),
        (exec_[0] - mw / 2, exec_[1]),
        "chunks + BM25 index",
        ((data[0] + exec_[0]) / 2, data[1] + 0.32),
    )
    # Execution → Optimisation (upper path: results)
    _arrow(
        ax,
        (exec_[0] + mw / 2, exec_[1] + 0.4),
        (opt[0] - mw / 2, opt[1] + 0.4),
        "fitness scores",
        ((exec_[0] + opt[0]) / 2, exec_[1] + 0.75),
    )
    # Optimisation → Execution (lower path: dispatch)
    _arrow(
        ax,
        (opt[0] - mw / 2, opt[1] - 0.4),
        (exec_[0] + mw / 2, exec_[1] - 0.4),
        "config vector x",
        ((exec_[0] + opt[0]) / 2, exec_[1] - 0.75),
        color=ARROW_GREY,
    )

    # ── Side layers ─────────────────────────────────────────────────────
    # Result Cache ↔ Execution (intercepts the dispatch→result cycle)
    _arrow(
        ax,
        (cache[0], cache[1] - sh / 2),
        (exec_[0], exec_[1] + mh / 2),
        "lookup / store",
        ((cache[0] + exec_[0]) / 2, (cache[1] + exec_[1]) / 2),
        color=ARROW_PURPLE, style="<->",
    )
    # Checkpoint ↔ Optimisation
    _arrow(
        ax,
        (ckpt[0], ckpt[1] - sh / 2),
        (opt[0], opt[1] + mh / 2),
        "save / restore",
        ((ckpt[0] + opt[0]) / 2, (ckpt[1] + opt[1]) / 2),
        color=ARROW_MAGENTA, style="<->",
    )

    # External dataset label above the Data Module.
    ax.text(
        data[0], data[1] + mh / 2 + 0.45,
        "HotpotQA · CUAD",
        ha="center", va="bottom",
        fontsize=10, style="italic", color="#555555",
    )

    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
