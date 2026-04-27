"""RAGAS Faithfulness evaluation using DeepInfra 70B judge.

Computes faithfulness scores for all Phase 16 held-out results using
meta-llama/Llama-3.3-70B-Instruct-Turbo (served by DeepInfra) as an
independent judge. Processes in batches of 10 with a 2s sleep between
batches, and checkpoints to ragas_faithfulness_partial.json after every
batch so an interrupted run loses at most one batch.

The generator under evaluation is Llama-3.1-8B on DeepInfra too, but at a
different parameter scale and instruction-tuning, so 70B-judging-8B still
provides meaningful signal against self-preference bias at the cost of
being same-provider.

DO NOT run this script automatically — run after held-out evaluation.

Run from project root:
    python scripts/run_ragas_faithfulness.py

Requires DEEPINFRA_API_KEY in .env.
Output: results/phase16/ragas_faithfulness.json
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

import openai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
HELD_OUT_PATH = Path("results/phase16/held_out_phase16.json")
HELD_OUT_RUNS_GENERAL = Path("results/phase16/held_out_phase16_general_runs")
HELD_OUT_RUNS_LEGAL = Path("results/phase16/held_out_phase16_legal_runs")

OUT_PATH = Path("results/phase16/ragas_faithfulness.json")
TEMP_PATH = Path("results/phase16/ragas_faithfulness_partial.json")

BATCH_SIZE = 10
SLEEP_BETWEEN_BATCHES = 2.0
N_SAMPLES_PER_SYSTEM = 50

TARGET_SYSTEMS: dict[str, list[str]] = {
    "general_qa": ["GA-P16-QA-S42", "P12-Best", "Naive-RAG", "Iterative-RAG"],
    "legal": ["GA-P16-Legal-S42", "P13-Best", "Naive-RAG", "Iterative-RAG"],
}

# (track, system_name) -> run_id directory under held_out_phase16_{track}_runs.
# Direct-LLM and Iterative-RAG do not persist a pipeline_results.jsonl and so
# cannot be scored — they have no entry here.
RUN_DIR_MAP: dict[tuple[str, str], str] = {
    ("general_qa", "Naive-RAG"): "naive_rag_held_out_general",
    ("general_qa", "Conservative-RAG"): "conservative_rag_held_out_general",
    ("general_qa", "P12-Best"): "p12_best_held_out_general",
    ("general_qa", "P13-Best-on-QA"): "p13_best_on_qa_held_out_general",
    ("general_qa", "GA-P16-QA-S42"): "ga_p16_qa_s42_held_out_general",
    ("general_qa", "GA-P16-QA-S123"): "ga_p16_qa_s123_held_out_general",
    ("general_qa", "GA-P16-QA-S7"): "ga_p16_qa_s7_held_out_general",
    ("general_qa", "GA-P16-Legal-S42-on-QA"): "ga_p16_legal_s42_on_qa_held_out_general",
    ("legal", "Naive-RAG"): "naive_rag_held_out_legal",
    ("legal", "Conservative-RAG"): "conservative_rag_held_out_legal",
    ("legal", "P12-Best-on-Legal"): "p12_best_on_legal_held_out_legal",
    ("legal", "P13-Best"): "p13_best_held_out_legal",
    ("legal", "GA-P16-Legal-S42"): "ga_p16_legal_s42_held_out_legal",
    ("legal", "GA-P16-QA-S42-on-Legal"): "ga_p16_qa_s42_on_legal_held_out_legal",
    ("legal", "GA-P16-QA-S123-on-Legal"): "ga_p16_qa_s123_on_legal_held_out_legal",
    ("legal", "GA-P16-QA-S7-on-Legal"): "ga_p16_qa_s7_on_legal_held_out_legal",
    ("general_qa", "Iterative-RAG"): "iterative_rag_held_out_general",
    ("legal", "Iterative-RAG"): "iterative_rag_held_out_legal",
}

# ── Judge client ───────────────────────────────────────────────────────────
# DeepInfra exposes an OpenAI-compatible endpoint, so we use the openai SDK
# with a pinned base_url and the DeepInfra API key.
judge_client = openai.OpenAI(
    api_key=os.environ["DEEPINFRA_API_KEY"],
    base_url="https://api.deepinfra.com/v1/openai",
)
judge_model = os.environ.get(
    "DEEPINFRA_JUDGE_MODEL_ID", "meta-llama/Llama-3.3-70B-Instruct-Turbo"
)


def _score_faithfulness(
    question: str,
    answer: str,
    context: str,
    client: openai.OpenAI,
    model: str,
) -> float:
    """Score faithfulness of an answer given context using LLM judge.

    Returns a score between 0.0 and 1.0.
    """
    prompt = (
        "You are evaluating the faithfulness of an answer to a question "
        "based on provided context. Faithfulness measures whether every "
        "claim in the answer can be inferred from the context.\n\n"
        f"Question: {question}\n\n"
        f"Context: {context}\n\n"
        f"Answer: {answer}\n\n"
        "Rate the faithfulness of the answer on a scale from 0 to 1, "
        "where 0 means completely unfaithful (makes claims not supported "
        "by context) and 1 means completely faithful (all claims supported "
        "by context). Respond with ONLY a number between 0 and 1."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )
    raw = response.choices[0].message.content.strip()
    try:
        score = float(raw)
        return max(0.0, min(1.0, score))
    except ValueError:
        logger.warning("Could not parse faithfulness score: %r", raw)
        return 0.0


def _build_context_from_chunks(chunks: list[dict]) -> str:
    """Build a single context string from retrieved chunk dicts."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "")
        parts.append(f"[DOC_{i}] {text}")
    return "\n\n".join(parts)


def _load_pipeline_results(runs_dir: Path, run_id: str) -> list[dict]:
    """Load pipeline_results.jsonl for a given run.

    Searches for the run_id directory under runs_dir and reads the
    pipeline_results.jsonl file inside it.
    """
    jsonl_path = runs_dir / run_id / "pipeline_results.jsonl"
    if not jsonl_path.exists():
        logger.warning("Pipeline results not found: %s", jsonl_path)
        return []
    results: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _collect_system_results(
    track: str, targets: list[str], runs_dir: Path
) -> list[tuple[str, list[dict] | None]]:
    """Resolve each target system to (name, pipeline_results) or (name, None).

    Looks up (track, name) in RUN_DIR_MAP; systems with no mapping (e.g.
    Direct-LLM, Iterative-RAG) yield None to be reported as unavailable.
    """
    collected: list[tuple[str, list[dict] | None]] = []
    for name in targets:
        run_id = RUN_DIR_MAP.get((track, name))
        if run_id is None:
            logger.warning(
                "No pipeline_results for '%s' (%s) — system kind does not "
                "persist retrieved context; will be reported as unavailable.",
                name, track,
            )
            collected.append((name, None))
            continue
        jsonl = runs_dir / run_id / "pipeline_results.jsonl"
        if not jsonl.exists():
            logger.warning(
                "pipeline_results.jsonl missing at %s for '%s' — marking "
                "unavailable.", jsonl, name,
            )
            collected.append((name, None))
            continue
        results = _load_pipeline_results(runs_dir, run_id)
        logger.info(
            "Loaded %d pipeline results for '%s' from %s",
            len(results), name, run_id,
        )
        collected.append((name, results))
    return collected


def _score_system(
    system_name: str,
    pipeline_results: list[dict],
    partial_scores: list[float] | None,
    on_batch: "callable | None" = None,
) -> list[float]:
    """Score faithfulness for first N_SAMPLES_PER_SYSTEM records of a system.

    Resumes from partial_scores if provided (incremental resume).
    Processes in batches of BATCH_SIZE with SLEEP_BETWEEN_BATCHES pause.
    """
    samples = pipeline_results[:N_SAMPLES_PER_SYSTEM]
    n_total = len(samples)

    if partial_scores is None:
        scores: list[float] = []
    else:
        scores = list(partial_scores)

    start_idx = len(scores)
    if start_idx >= n_total:
        logger.info(
            "System '%s' already complete (%d/%d scores)",
            system_name, len(scores), n_total,
        )
        return scores

    logger.info(
        "Scoring '%s': resuming from record %d/%d",
        system_name, start_idx, n_total,
    )

    remaining = samples[start_idx:]
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start : batch_start + BATCH_SIZE]
        for record in batch:
            question = record.get("question", "")
            answer = record.get("answer", "")
            chunks = record.get("retrieved_chunks", [])
            context = _build_context_from_chunks(chunks)

            score = _score_faithfulness(
                question=question,
                answer=answer,
                context=context,
                client=judge_client,
                model=judge_model,
            )
            scores.append(score)

        abs_idx = start_idx + batch_start + len(batch)
        logger.info(
            "  '%s': scored %d/%d records",
            system_name, abs_idx, n_total,
        )
        if on_batch is not None:
            on_batch(list(scores))

        # Sleep between batches (not after the last one)
        if batch_start + BATCH_SIZE < len(remaining):
            time.sleep(SLEEP_BETWEEN_BATCHES)

    return scores


def _load_partial() -> dict | None:
    """Load partial results from TEMP_PATH if it exists."""
    if TEMP_PATH.exists():
        logger.info("Resuming from partial results: %s", TEMP_PATH)
        return json.loads(TEMP_PATH.read_text(encoding="utf-8"))
    return None


def _save_partial(data: dict) -> None:
    """Save partial results to TEMP_PATH for incremental resume."""
    TEMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEMP_PATH.write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Saved partial results to %s", TEMP_PATH)


def _get_partial_scores(
    partial: dict | None, track: str, system_name: str
) -> list[float] | None:
    """Extract partial scores for a specific system from saved state."""
    if partial is None:
        return None
    track_data = partial.get(track, {})
    for sys_entry in track_data.get("systems", []):
        if sys_entry.get("name") == system_name:
            return sys_entry.get("scores")
    return None


def main() -> None:
    """Run RAGAS faithfulness evaluation on Phase 16 held-out results."""
    # ── Load held-out results ──────────────────────────────────────────────
    if not HELD_OUT_PATH.exists():
        logger.error(
            "Held-out results not found: %s. "
            "Run held-out evaluation first.",
            HELD_OUT_PATH,
        )
        sys.exit(1)

    logger.info("Held-out results path confirmed: %s", HELD_OUT_PATH)

    # ── Load partial results for resume ────────────────────────────────────
    partial = _load_partial()

    # ── Initialise output structure ────────────────────────────────────────
    output: dict = {
        "judge_model": judge_model,
        "n_samples_per_system": N_SAMPLES_PER_SYSTEM,
        "general_qa": {"systems": []},
        "legal": {"systems": []},
    }

    runs_by_track = {
        "general_qa": HELD_OUT_RUNS_GENERAL,
        "legal": HELD_OUT_RUNS_LEGAL,
    }

    for track in ("general_qa", "legal"):
        print(f"\n{'=' * 65}")
        print(f"RAGAS Faithfulness — {track}")
        print(f"{'=' * 65}")

        targets = TARGET_SYSTEMS[track]
        collected = _collect_system_results(
            track, targets, runs_by_track[track]
        )

        for system_name, pipeline_results in collected:
            if pipeline_results is None:
                output[track]["systems"].append({
                    "name": system_name,
                    "mean_faithfulness": None,
                    "scores": [],
                    "note": "unavailable: no persisted pipeline_results.jsonl",
                })
                print(
                    f"  {system_name}: UNAVAILABLE "
                    f"(no persisted retrieved_chunks)"
                )
                _save_partial(output)
                continue

            partial_scores = _get_partial_scores(
                partial, track, system_name
            )

            # Add a placeholder entry so per-batch checkpoints mutate in place.
            entry: dict = {
                "name": system_name,
                "mean_faithfulness": None,
                "scores": list(partial_scores) if partial_scores else [],
            }
            output[track]["systems"].append(entry)

            def _checkpoint(new_scores: list[float]) -> None:
                entry["scores"] = new_scores
                entry["mean_faithfulness"] = (
                    round(sum(new_scores) / len(new_scores), 4)
                    if new_scores else 0.0
                )
                _save_partial(output)

            scores = _score_system(
                system_name, pipeline_results, partial_scores,
                on_batch=_checkpoint,
            )
            mean_faith = sum(scores) / len(scores) if scores else 0.0

            entry["mean_faithfulness"] = round(mean_faith, 4)
            entry["scores"] = scores
            print(
                f"  {system_name}: mean_faithfulness={mean_faith:.4f} "
                f"({len(scores)} samples)"
            )

            _save_partial(output)

    # ── Finalise ───────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved final results to {OUT_PATH}")

    # Clean up partial file
    if TEMP_PATH.exists():
        TEMP_PATH.unlink()
        logger.info("Removed partial file %s", TEMP_PATH)

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"{'System':<32} | {'Track':<12} | {'Faithfulness':>12}")
    print("-" * 65)
    for track_key, track_label in (
        ("general_qa", "General-QA"), ("legal", "Legal")
    ):
        for sys_entry in output[track_key]["systems"]:
            mf = sys_entry["mean_faithfulness"]
            mf_str = f"{mf:>12.4f}" if isinstance(mf, float) else f"{'N/A':>12}"
            print(
                f"{sys_entry['name']:<32} | {track_label:<12} | {mf_str}"
            )
    print(f"{'=' * 65}")
    print(f"Judge model: {judge_model}")
    print(f"Samples per system: {N_SAMPLES_PER_SYSTEM}")


if __name__ == "__main__":
    main()

    # Force exit — httpx connection pool in the groq client holds
    # non-daemon threads that prevent natural process termination.
    sys.exit(0)
