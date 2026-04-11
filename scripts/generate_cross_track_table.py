"""Generate the master cross-track comparison table as CSV and JSON.

Reads all available Phase 12, Phase 13, Phase 14, and ablation result
files. Missing files are skipped with a warning so this script can be
run at any point — even before legal_baselines.json and
cross_track_transfer.json exist.

Run from project root:
    python scripts/generate_cross_track_table.py

Outputs:
    results/phase14/cross_track_comparison.csv
    results/phase14/cross_track_summary.json
"""

import csv
import json
from pathlib import Path

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_DIR = Path("results/phase14")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "cross_track_comparison.csv"
JSON_PATH = OUT_DIR / "cross_track_summary.json"

CSV_COLUMNS = [
    "System",
    "Track",
    "Fitness",
    "EM",
    "F1",
    "CitP",
    "CitR",
    "CitF1",
    "RetR@5",
    "InputTokens",
    "LatencyMs",
]


def _try_load(path: Path) -> dict | list | None:
    """Load a JSON file, printing a warning and returning None if missing."""
    if not path.exists():
        print(f"WARNING: {path} not found — skipping")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics_to_row(
    name: str,
    track: str,
    fitness: float | None,
    metrics: dict,
) -> dict:
    """Build a unified row dict from a metrics dict."""
    return {
        "system": name,
        "track": track,
        "fitness": fitness,
        "em": metrics.get("mean_em"),
        "f1": metrics.get("mean_f1"),
        "cit_p": metrics.get("mean_citation_precision"),
        "cit_r": metrics.get("mean_citation_recall"),
        "cit_f1": metrics.get("mean_citation_f1"),
        "ret_r_at_k": metrics.get("mean_retrieval_recall_at_k"),
        "input_tokens": metrics.get("mean_input_tokens"),
        "latency_ms": metrics.get("mean_latency_ms"),
    }


def _components_to_row(
    name: str,
    track: str,
    components: dict,
) -> dict:
    """Build a row from GA best_components (partial metrics only)."""
    return {
        "system": name,
        "track": track,
        "fitness": components.get("fitness"),
        "em": None,
        "f1": components.get("q_norm"),
        "cit_p": None,
        "cit_r": None,
        "cit_f1": components.get("g_norm"),
        "ret_r_at_k": components.get("r_norm"),
        "input_tokens": None,
        "latency_ms": None,
    }


rows: list[dict] = []

# ── General-QA baselines ──────────────────────────────────────────────────────
data = _try_load(Path("results/ablations/baseline_comparison.json"))
if data is not None:
    for s in data["systems"]:
        rows.append(_metrics_to_row(s["name"], "general", s["fitness"], s["metrics"]))
    print(f"Loaded {len(data['systems'])} General-QA baseline rows")

# ── General-QA iterative RAG ──────────────────────────────────────────────────
data = _try_load(Path("results/ablations/iterative_rag.json"))
if data is not None:
    rows.append(
        _metrics_to_row("Iterative RAG", "general", data["fitness"], data["metrics"])
    )
    print("Loaded General-QA iterative RAG row")

# ── Phase 12 GA best (from summary, partial metrics) ─────────────────────────
data = _try_load(Path("results/phase12/general_qa/phase12_summary.json"))
if data is not None:
    best_gen = max(data, key=lambda r: r["best_fitness"])
    rows.append(_components_to_row(
        "GA Phase 12 best", "general", best_gen["best_components"]
    ))
    print(f"Loaded Phase 12 best from gen {best_gen['generation_idx']}")

# ── Legal baselines ───────────────────────────────────────────────────────────
data = _try_load(Path("results/phase14/legal_baselines.json"))
if data is not None:
    for s in data["systems"]:
        rows.append(_metrics_to_row(s["name"], "legal", s["fitness"], s["metrics"]))
    print(f"Loaded {len(data['systems'])} legal baseline rows")

# ── Legal iterative RAG ───────────────────────────────────────────────────────
data = _try_load(Path("results/phase13/iterative_rag_legal.json"))
if data is not None:
    rows.append(_metrics_to_row(
        "Iterative RAG Legal", "legal", data["fitness"], data["metrics"]
    ))
    print("Loaded legal iterative RAG row")

# ── Phase 13 GA best (from summary, partial metrics) ─────────────────────────
data = _try_load(Path("results/phase13/legal_qa/phase13_summary.json"))
if data is not None:
    best_gen = max(data, key=lambda r: r["best_fitness"])
    rows.append(_components_to_row(
        "GA Phase 13 best", "legal", best_gen["best_components"]
    ))
    print(f"Loaded Phase 13 best from gen {best_gen['generation_idx']}")

# ── Cross-track transfer ──────────────────────────────────────────────────────
data = _try_load(Path("results/phase14/cross_track_transfer.json"))
if data is not None:
    for t in data["transfers"]:
        track = t["evaluated_on"]
        name = f"{t['config_source']} → {track}"
        rows.append(_metrics_to_row(name, track, t["fitness"], t["metrics"]))
    print(f"Loaded {len(data['transfers'])} cross-track transfer rows")

# ── Write CSV ─────────────────────────────────────────────────────────────────
def _fmt(val: float | None) -> str:
    if val is None:
        return ""
    return f"{val:.6g}"


with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        writer.writerow([
            row["system"],
            row["track"],
            _fmt(row["fitness"]),
            _fmt(row["em"]),
            _fmt(row["f1"]),
            _fmt(row["cit_p"]),
            _fmt(row["cit_r"]),
            _fmt(row["cit_f1"]),
            _fmt(row["ret_r_at_k"]),
            _fmt(row["input_tokens"]),
            _fmt(row["latency_ms"]),
        ])

# ── Write JSON ────────────────────────────────────────────────────────────────
JSON_PATH.write_text(
    json.dumps({"rows": rows}, indent=2) + "\n",
    encoding="utf-8",
)

print(f"\nTotal rows: {len(rows)}")
print(f"CSV  → {CSV_PATH}")
print(f"JSON → {JSON_PATH}")
