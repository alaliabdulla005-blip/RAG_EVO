"""CUAD legal data loader for the RAG-Evo Legal QA track.

Downloads and processes the CUAD-QA dataset from HuggingFace.
Produces QARecord and CorpusDocument objects in the same format
as hotpotqa_loader.py, so the evaluation harness can be used
without modification.

CUAD structure (SQuAD-formatted):
  id      — unique question identifier
  title   — contract filename (used as doc_id)
  context — full contract text (can be millions of characters)
  question — one of 41 clause-category questions
  answers — {'text': [...], 'answer_start': [...]}
            Empty lists for unanswerable questions.

Key differences from HotpotQA:
  - No multi-hop supporting facts: supporting_facts is always [].
  - source_doc_ids has exactly one entry (the contract doc_id).
  - Contexts are truncated to _MAX_CONTEXT_CHARS before chunking
    because legal contracts vastly exceed the BM25 chunker's
    memory and are dominated by boilerplate outside the first
    2 000 characters. This limits retrieval recall for late clauses
    but is required for practical operation.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

from datasets import load_dataset

from rag_evo.data.schemas import CorpusDocument, QARecord, SupportingFact

logger = logging.getLogger(__name__)

# Contracts can be millions of characters; truncate to this limit before
# chunking to prevent memory exhaustion and extremely long index builds.
_MAX_CONTEXT_CHARS = 2000

# theatticusproject/cuad-qa uses a legacy dataset script that is no longer
# supported by the datasets library. chenghao/cuad_qa is the same data in
# standard Parquet format with an identical SQuAD-style schema.
_DATASET_ID = "chenghao/cuad_qa"
_SPLIT = "test"


def _make_doc_id(title: str) -> str:
    """Derive a stable document ID from a contract title.

    Args:
        title: Contract filename or title as it appears in CUAD.

    Returns:
        Lowercase, underscore-separated identifier string.
    """
    return title.strip().replace(" ", "_").lower()


def load_cuad(
    cache_dir: Path | None = None,
) -> tuple[list[QARecord], list[CorpusDocument]]:
    """Load and process CUAD-QA from HuggingFace.

    Downloads the test split of theatticusproject/cuad-qa.
    Skips unanswerable questions (empty answers field).
    Truncates each contract context to _MAX_CONTEXT_CHARS.

    Args:
        cache_dir: HuggingFace dataset cache directory.
            If None, uses the default HuggingFace cache.

    Returns:
        Tuple of (qa_records, corpus_documents) where:
            - qa_records contains only answerable questions
            - corpus_documents is deduplicated by doc_id and
              contains truncated contract texts
    """
    cache_str = str(cache_dir) if cache_dir is not None else None
    ds = load_dataset(_DATASET_ID, split=_SPLIT, cache_dir=cache_str)

    qa_records: list[QARecord] = []
    corpus_by_id: dict[str, CorpusDocument] = {}

    for example in ds:
        answers = example["answers"]
        texts = answers.get("text", [])

        # Skip unanswerable questions
        if not texts or not texts[0].strip():
            continue

        gold_answer = texts[0].strip()
        doc_id = _make_doc_id(example["title"])

        qa_records.append(
            QARecord(
                record_id=example["id"],
                question=example["question"],
                gold_answer=gold_answer,
                # CUAD has no HotpotQA-style sentence-level supporting
                # facts — use an empty list for compatibility.
                supporting_facts=[],
                source_doc_ids=[doc_id],
                split=_SPLIT,
            )
        )

        if doc_id not in corpus_by_id:
            # Truncate to _MAX_CONTEXT_CHARS — see module docstring.
            truncated = example["context"][:_MAX_CONTEXT_CHARS]
            corpus_by_id[doc_id] = CorpusDocument(
                doc_id=doc_id,
                title=example["title"],
                # Wrap the whole (truncated) contract as one element so
                # the CorpusDocument structure matches HotpotQA format.
                sentences=[truncated],
                text=truncated,
            )

    corpus_documents = list(corpus_by_id.values())

    logger.info(
        "Loaded CUAD %s: %d answerable QA records, %d unique contracts.",
        _SPLIT,
        len(qa_records),
        len(corpus_documents),
    )
    return qa_records, corpus_documents


def build_legal_debug_subset(
    output_dir: Path = Path("data/processed/legal_debug"),
    n: int = 50,
    cache_dir: Path | None = None,
) -> tuple[list[QARecord], list[CorpusDocument]]:
    """Build and save the legal debug subset to disk.

    Downloads CUAD, takes the first n answerable records, and saves
    them in the same JSONL format as hotpotqa_loader.save_records().

    Args:
        output_dir: Destination directory for qa_records.jsonl and
            corpus.jsonl.
        n: Number of records in the debug subset. Default 50.
        cache_dir: HuggingFace dataset cache directory.

    Returns:
        Tuple of (debug_qa_records, corpus_documents).
    """
    qa_records, corpus_documents = load_cuad(cache_dir=cache_dir)
    debug_records = qa_records[:n]
    save_records(debug_records, corpus_documents, output_dir)
    logger.info(
        "Saved %d legal debug records and %d corpus docs to %s",
        len(debug_records),
        len(corpus_documents),
        output_dir,
    )
    return debug_records, corpus_documents


def save_records(
    qa_records: list[QARecord],
    corpus_documents: list[CorpusDocument],
    output_dir: Path,
) -> None:
    """Save QA records and corpus documents to JSONL files.

    Creates:
        {output_dir}/qa_records.jsonl  — one QARecord per line
        {output_dir}/corpus.jsonl      — one CorpusDocument per line

    Args:
        qa_records: List of QARecord objects to save.
        corpus_documents: List of CorpusDocument objects to save.
        output_dir: Directory to write the files into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "qa_records.jsonl").open("w", encoding="utf-8") as f:
        for record in qa_records:
            f.write(json.dumps(asdict(record)) + "\n")

    with (output_dir / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for doc in corpus_documents:
            f.write(json.dumps(asdict(doc)) + "\n")


def load_legal_records(
    output_dir: Path,
) -> tuple[list[QARecord], list[CorpusDocument]]:
    """Load previously saved legal QA records and corpus from JSONL files.

    Drop-in replacement for hotpotqa_loader.load_records().

    Args:
        output_dir: Directory containing qa_records.jsonl and corpus.jsonl.

    Returns:
        Tuple of (qa_records, corpus_documents).

    Raises:
        FileNotFoundError: If either file does not exist.
    """
    qa_path = output_dir / "qa_records.jsonl"
    corpus_path = output_dir / "corpus.jsonl"

    if not qa_path.exists():
        raise FileNotFoundError(f"QA records file not found: {qa_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    qa_records: list[QARecord] = []
    with qa_path.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            # supporting_facts is always [] for CUAD but deserialise
            # generically so the format stays compatible with HotpotQA.
            d["supporting_facts"] = [
                SupportingFact(**sf) for sf in d["supporting_facts"]
            ]
            qa_records.append(QARecord(**d))

    corpus_documents: list[CorpusDocument] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            corpus_documents.append(CorpusDocument(**json.loads(line)))

    return qa_records, corpus_documents
