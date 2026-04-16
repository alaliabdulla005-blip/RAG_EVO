"""Pre-build dense FAISS indexes for Phase 16 GA experiments.

Builds indexes for all combinations of:
  chunk_sizes: {256, 384, 448, 512}
  overlaps:    {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}
  corpora:     {general_qa, legal}

Total: 4 x 6 x 2 = 48 indexes.

Skips any index that already exists on disk.

DO NOT run this script automatically — run it once before GA starts.

Run from project root:
    python scripts/build_dense_indexes.py

Indexes saved to: data/indexes/dense_{chunk_size}_{overlap}_{track}/
"""

import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from rag_evo.data.hotpotqa_loader import load_records  # noqa: E402
from rag_evo.data.legal_loader import load_legal_records  # noqa: E402
from rag_evo.retrieval.chunker import chunk_corpus  # noqa: E402
from rag_evo.retrieval.dense_retriever import build_and_save_dense_index  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOTPOTQA_DIR = PROJECT_ROOT / "data" / "processed" / "hotpotqa_debug"
LEGAL_DIR = PROJECT_ROOT / "data" / "processed" / "legal_debug"
INDEX_ROOT = PROJECT_ROOT / "data" / "indexes"

CHUNK_SIZES = [256, 384, 448, 512]
OVERLAPS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def main() -> None:
    logger.info("Loading General-QA corpus from %s", HOTPOTQA_DIR)
    _, general_corpus = load_records(HOTPOTQA_DIR)
    logger.info("General-QA corpus: %d documents", len(general_corpus))

    logger.info("Loading Legal corpus from %s", LEGAL_DIR)
    _, legal_corpus = load_legal_records(LEGAL_DIR)
    logger.info("Legal corpus: %d documents", len(legal_corpus))

    corpora = {
        "general_qa": general_corpus,
        "legal": legal_corpus,
    }

    # Build job list
    jobs: list[tuple[int, float, str]] = []
    for chunk_size in CHUNK_SIZES:
        for overlap in OVERLAPS:
            for track in corpora:
                jobs.append((chunk_size, overlap, track))

    total = len(jobs)
    built = 0
    skipped = 0
    total_start = time.perf_counter()

    for i, (chunk_size, overlap, track) in enumerate(jobs, start=1):
        index_dir = INDEX_ROOT / f"dense_{chunk_size}_{overlap}_{track}"

        # Skip if already built (check for the index file)
        if index_dir.exists() and any(index_dir.iterdir()):
            print(
                f"Skipping (exists) {i}/{total}: "
                f"chunk_size={chunk_size} overlap={overlap} "
                f"track={track}"
            )
            skipped += 1
            continue

        print(
            f"Building {i}/{total}: "
            f"chunk_size={chunk_size} overlap={overlap} "
            f"track={track}..."
        )

        step_start = time.perf_counter()
        corpus_docs = corpora[track]
        chunks = chunk_corpus(
            corpus_docs, chunk_size=chunk_size, overlap=overlap
        )
        logger.info(
            "Chunked %d documents into %d chunks",
            len(corpus_docs),
            len(chunks),
        )

        build_and_save_dense_index(chunks, index_dir)
        elapsed = time.perf_counter() - step_start

        logger.info("Saved index to %s  (%.1fs)", index_dir, elapsed)
        built += 1

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'=' * 60}")
    print(
        f"Done in {total_elapsed:.1f}s: "
        f"{built} built, {skipped} skipped, {total} total"
    )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
    sys.exit(0)
