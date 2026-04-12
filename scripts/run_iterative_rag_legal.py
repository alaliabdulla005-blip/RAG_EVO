"""Legal iterative RAG baseline — 2-round adaptive retrieval on development records.

Algorithm mirrors run_iterative_rag.py but operates on legal QA data.
Per-record detail is logged to support Phase 14 error analysis.

Round 1 : BM25 retrieve (top_k=10) → chain-of-thought answer
Critique : Ask LLM whether the answer is complete; if not, get a
           refined query
Round 2  : Retrieve with refined query (top_k=10), merge chunks
           (deduplicated by text), generate final answer

Fixed config: chunk_size=384, overlap=0.3, temperature=0.0.
Evaluates records 0-49 (development set).

DO NOT run this script — run it manually after Phase 13 GA completes.

Run from project root:
    python scripts/run_iterative_rag_legal.py

Requires INFERENCE_PROVIDER and matching API key in .env.
"""

import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

from rag_evo.data.legal_loader import load_legal_records
from rag_evo.evaluation.harness import EvaluationHarness
from rag_evo.evaluation.iterative_rag import run_iterative_rag
from rag_evo.ga.fitness import FitnessWeights
from rag_evo.generation.wrapper import GeneratorWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

OUT_PATH = Path("results/phase13/iterative_rag_legal.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Loading legal debug subset...")
all_records, corpus_documents = load_legal_records(Path("data/processed/legal_debug"))
# Slice to development set (records 0-49) for backward-compatible results
dev_records = all_records[:50]
print(f"Using {len(dev_records)} records, {len(corpus_documents)} corpus docs")

wrapper = GeneratorWrapper()
harness = EvaluationHarness(dev_records)

# Intentionally uses FitnessWeights() (general weights), not legal_fitness_weights(),
# to match the original evaluation and preserve backward compatibility.
summary = run_iterative_rag(
    qa_records=dev_records,
    corpus_documents=corpus_documents,
    config={},  # all defaults: chunk=384, overlap=0.3, top_k=10, temp=0.0
    wrapper=wrapper,
    harness=harness,
    cache_dir=Path("data/cache/iterative_rag_legal"),
    fitness_weights=FitnessWeights(),
)

m = summary["metrics"]
n_round2 = summary["n_round2_triggered"]
print(f"\n{'=' * 60}")
print(f"Legal Iterative RAG: fitness={summary['fitness']:.4f}")
print(f"  EM={m['mean_em']:.4f} F1={m['mean_f1']:.4f}")
print(
    f"  CitP={m['mean_citation_precision']:.4f} "
    f"CitR={m['mean_citation_recall']:.4f} "
    f"CitF1={m['mean_citation_f1']:.4f}"
)
print(
    f"  RetR@k={m['mean_retrieval_recall_at_k']:.4f} "
    f"Tokens={m['mean_input_tokens']:.0f} "
    f"Lat={m['mean_latency_ms']:.0f}ms"
)
print(f"  Round 2 triggered: {n_round2}/{len(dev_records)} records")

# Override system name to match original output format
summary["system"] = "iterative_rag_legal"

OUT_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"\nSaved to {OUT_PATH}")

# Force exit — httpx connection pool in the openai/groq client holds
# non-daemon threads that prevent natural process termination.
sys.exit(0)
