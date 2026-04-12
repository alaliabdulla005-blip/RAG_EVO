"""Iterative RAG baseline — 2-round adaptive retrieval on development records.

Algorithm:
  Round 1 : BM25 retrieve (top_k=10) → chain-of-thought answer
  Critique : Ask LLM whether the answer is complete; if not, get a
             refined query
  Round 2  : Retrieve with refined query (top_k=10), merge chunks
             (deduplicated by text), generate final answer

Fixed config: chunk_size=384, overlap=0.3, temperature=0.0.
Evaluates records 0-49 (development set).

DO NOT run this script until the Phase 12 GA has started — it shares
the SambaNova API quota and should not compete with the GA.

Run from project root:
    python scripts/run_iterative_rag.py

Requires INFERENCE_PROVIDER and matching API key in .env.
"""

import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

from rag_evo.data.hotpotqa_loader import load_records
from rag_evo.evaluation.harness import EvaluationHarness
from rag_evo.evaluation.iterative_rag import run_iterative_rag
from rag_evo.ga.fitness import FitnessWeights
from rag_evo.generation.wrapper import GeneratorWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

OUT_PATH = Path("results/ablations/iterative_rag.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Loading HotpotQA debug subset...")
all_records, corpus_documents = load_records(Path("data/processed/hotpotqa_debug"))
# Slice to development set (records 0-49) for backward-compatible results
dev_records = all_records[:50]
print(f"Using {len(dev_records)} records, {len(corpus_documents)} corpus docs")

wrapper = GeneratorWrapper()
harness = EvaluationHarness(dev_records)

summary = run_iterative_rag(
    qa_records=dev_records,
    corpus_documents=corpus_documents,
    config={},  # all defaults: chunk=384, overlap=0.3, top_k=10, temp=0.0
    wrapper=wrapper,
    harness=harness,
    cache_dir=Path("data/cache/iterative_rag"),
    fitness_weights=FitnessWeights(),
)

m = summary["metrics"]
print(f"\n{'=' * 60}")
print(f"Iterative RAG: fitness={summary['fitness']:.4f}")
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

# Save without per_record — original output did not include it
save_summary = {k: v for k, v in summary.items() if k != "per_record"}
OUT_PATH.write_text(json.dumps(save_summary, indent=2) + "\n", encoding="utf-8")
print(f"\nSaved to {OUT_PATH}")

# Force exit — httpx connection pool in the openai/groq client holds
# non-daemon threads that prevent natural process termination.
sys.exit(0)
