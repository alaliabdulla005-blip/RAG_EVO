"""RAGAS Faithfulness evaluation using Groq 70B judge.

Computes faithfulness scores for all Phase 16 held-out results using
llama-3.3-70b-versatile as an independent judge to reduce self-preference
bias. Processes in batches of 10 with 2s sleep between batches.
Saves incrementally and resumes from the last saved batch.

DO NOT run this script automatically — run after held-out evaluation.

Run from project root:
    python scripts/run_ragas_faithfulness.py

Requires GROQ_API_KEY in .env.
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

import groq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
HELD_OUT_PATH = Path("results/phase16/held_out_phase16.json")
HELD_OUT_RUNS_GENERAL = Path("results/phase16/held_out_general_runs")
HELD_OUT_RUNS_LEGAL = Path("results/phase16/held_out_legal_runs")

OUT_PATH = Path("results/phase16/ragas_faithfulness.json")
TEMP_PATH = Path("results/phase16/ragas_faithfulness_partial.json")

BATCH_SIZE = 10
SLEEP_BETWEEN_BATCHES = 2.0
N_SAMPLES_PER_SYSTEM = 50

# ── Judge client ───────────────────────────────────────────────────────────
judge_client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
judge_model = os.environ.get(
    "GROQ_JUDGE_MODEL_ID", "llama-3.3-70b-versatile"
)


def _score_faithfulness(
    question: str,
    answer: str,
    context: str,
    client: groq.Groq,
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


def _find_run_id_for_system(
    system_name: str, runs_dir: Path
) -> str | None:
    """Derive the run_id directory name from the system name.

    The held-out evaluation scripts use a naming convention for run_ids.
    We look for any directory in runs_dir that contains pipeline_results.jsonl
    and match by checking the config.json inside against the system name.

    Falls back to scanning all subdirectories and matching system name
    in run_summary.json or config.json.
    """
    if not runs_dir.exists():
        return None
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        jsonl = run_dir / "pipeline_results.jsonl"
        if jsonl.exists():
            return run_dir.name
    return None


def _collect_system_results(
    systems: list[dict], runs_dir: Path
) -> list[tuple[str, list[dict]]]:
    """For each system, find and load its pipeline results.

    Returns list of (system_name, pipeline_results) tuples.
    Only includes systems for which pipeline_results.jsonl was found.
    """
    collected: list[tuple[str, list[dict]]] = []
    # Get all run directories sorted
    if not runs_dir.exists():
        logger.warning("Runs directory not found: %s", runs_dir)
        return collected

    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    # Match systems to run directories in order — the held-out evaluation
    # scripts create one run directory per system in evaluation order.
    for i, system in enumerate(systems):
        name = system["name"]
        if i < len(run_dirs):
            run_dir = run_dirs[i]
            jsonl = run_dir / "pipeline_results.jsonl"
            if jsonl.exists():
                results = _load_pipeline_results(runs_dir, run_dir.name)
                collected.append((name, results))
                logger.info(
                    "Loaded %d pipeline results for '%s' from %s",
                    len(results), name, run_dir.name,
                )
            else:
                logger.warning(
                    "No pipeline_results.jsonl in %s for system '%s'",
                    run_dir, name,
                )
        else:
            logger.warning(
                "No run directory found for system '%s' (index %d)",
                name, i,
            )
    return collected


def _score_system(
    system_name: str,
    pipeline_results: list[dict],
    partial_scores: list[float] | None,
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

    held_out = json.loads(HELD_OUT_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded held-out results from %s", HELD_OUT_PATH)

    general_systems = held_out.get("general_qa", {}).get("systems", [])
    legal_systems = held_out.get("legal", {}).get("systems", [])

    print(f"General-QA systems: {len(general_systems)}")
    print(f"Legal systems: {len(legal_systems)}")

    # ── Load partial results for resume ────────────────────────────────────
    partial = _load_partial()

    # ── Initialise output structure ────────────────────────────────────────
    output: dict = {
        "judge_model": judge_model,
        "n_samples_per_system": N_SAMPLES_PER_SYSTEM,
        "general_qa": {"systems": []},
        "legal": {"systems": []},
    }

    # ── General-QA track ───────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("RAGAS Faithfulness — General-QA track")
    print(f"{'=' * 65}")

    general_collected = _collect_system_results(
        general_systems, HELD_OUT_RUNS_GENERAL
    )

    for system_name, pipeline_results in general_collected:
        partial_scores = _get_partial_scores(
            partial, "general_qa", system_name
        )
        scores = _score_system(system_name, pipeline_results, partial_scores)
        mean_faith = sum(scores) / len(scores) if scores else 0.0

        output["general_qa"]["systems"].append({
            "name": system_name,
            "mean_faithfulness": round(mean_faith, 4),
            "scores": scores,
        })
        print(
            f"  {system_name}: mean_faithfulness={mean_faith:.4f} "
            f"({len(scores)} samples)"
        )

        # Incremental save after each system
        _save_partial(output)

    # ── Legal track ────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("RAGAS Faithfulness — Legal track")
    print(f"{'=' * 65}")

    legal_collected = _collect_system_results(
        legal_systems, HELD_OUT_RUNS_LEGAL
    )

    for system_name, pipeline_results in legal_collected:
        partial_scores = _get_partial_scores(
            partial, "legal", system_name
        )
        scores = _score_system(system_name, pipeline_results, partial_scores)
        mean_faith = sum(scores) / len(scores) if scores else 0.0

        output["legal"]["systems"].append({
            "name": system_name,
            "mean_faithfulness": round(mean_faith, 4),
            "scores": scores,
        })
        print(
            f"  {system_name}: mean_faithfulness={mean_faith:.4f} "
            f"({len(scores)} samples)"
        )

        # Incremental save after each system
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
    for sys_entry in output["general_qa"]["systems"]:
        print(
            f"{sys_entry['name']:<32} | {'General-QA':<12} "
            f"| {sys_entry['mean_faithfulness']:>12.4f}"
        )
    for sys_entry in output["legal"]["systems"]:
        print(
            f"{sys_entry['name']:<32} | {'Legal':<12} "
            f"| {sys_entry['mean_faithfulness']:>12.4f}"
        )
    print(f"{'=' * 65}")
    print(f"Judge model: {judge_model}")
    print(f"Samples per system: {N_SAMPLES_PER_SYSTEM}")


if __name__ == "__main__":
    main()

    # Force exit — httpx connection pool in the groq client holds
    # non-daemon threads that prevent natural process termination.
    sys.exit(0)
