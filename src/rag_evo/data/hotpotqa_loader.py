"""HotpotQA data loader for the RAG-Evo General QA track.

Downloads and processes the HotpotQA distractor setting dataset.
Produces QARecord and CorpusDocument objects conforming to schemas.py.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

from datasets import load_dataset

from rag_evo.data.schemas import CorpusDocument, QARecord, SupportingFact

logger = logging.getLogger(__name__)

_VALID_SPLITS = {"train", "validation"}


def make_doc_id(title: str) -> str:
    """Derive a stable document ID from a Wikipedia article title.

    Args:
        title: Wikipedia article title as it appears in HotpotQA.

    Returns:
        Lowercase, underscore-separated identifier string.

    Examples:
        >>> make_doc_id("Eiffel Tower")
        'eiffel_tower'
        >>> make_doc_id("  Barack Obama  ")
        'barack_obama'
    """
    return title.strip().replace(" ", "_").lower()


def load_hotpotqa(
    split: str = "validation",
    cache_dir: Path | None = None,
    max_records: int | None = None,
) -> tuple[list[QARecord], list[CorpusDocument]]:
    """Load and process HotpotQA distractor setting for the given split.

    Downloads via HuggingFace datasets library (hotpot_qa, distractor config).
    Processes each example into QARecord and CorpusDocument objects.

    Args:
        split: Dataset split to load. Use "validation" for development,
            "train" for full experiments.
        cache_dir: Directory to cache the downloaded dataset.
            If None, uses HuggingFace default cache.
        max_records: If set, return only the first max_records examples.
            Always takes the first N — never random.

    Returns:
        Tuple of (qa_records, corpus_documents) where corpus_documents is
        deduplicated by doc_id.

    Raises:
        ValueError: If split is not "train" or "validation".
    """
    if split not in _VALID_SPLITS:
        raise ValueError(
            f"split must be one of {_VALID_SPLITS}, got {split!r}"
        )

    cache_str = str(cache_dir) if cache_dir is not None else None
    ds = load_dataset(
        "hotpot_qa", "distractor", split=split, cache_dir=cache_str
    )

    qa_records: list[QARecord] = []
    corpus_by_id: dict[str, CorpusDocument] = {}

    for example in ds:
        # --- Build supporting facts ---
        sf_titles = example["supporting_facts"]["title"]
        sf_sent_ids = example["supporting_facts"]["sent_id"]
        supporting_facts = [
            SupportingFact(doc_title=t, sent_idx=i)
            for t, i in zip(sf_titles, sf_sent_ids)
        ]

        # --- Derive source_doc_ids (deduplicated, order-preserving) ---
        seen: dict[str, None] = {}
        for sf in supporting_facts:
            did = make_doc_id(sf.doc_title)
            seen[did] = None
        source_doc_ids = list(seen.keys())

        qa_records.append(
            QARecord(
                record_id=example["id"],
                question=example["question"],
                gold_answer=example["answer"],
                supporting_facts=supporting_facts,
                source_doc_ids=source_doc_ids,
                split=split,
            )
        )

        # --- Build corpus documents (deduplicated by doc_id) ---
        for title, sentences in zip(
            example["context"]["title"],
            example["context"]["sentences"],
        ):
            did = make_doc_id(title)
            if did not in corpus_by_id:
                corpus_by_id[did] = CorpusDocument(
                    doc_id=did,
                    title=title,
                    sentences=list(sentences),
                    text=" ".join(sentences),
                )

    corpus_documents = list(corpus_by_id.values())

    logger.info(
        "Loaded split=%s: %d QA records, %d unique corpus documents.",
        split,
        len(qa_records),
        len(corpus_documents),
    )

    if max_records is not None:
        qa_records = qa_records[:max_records]

    return qa_records, corpus_documents


def build_debug_subset(
    qa_records: list[QARecord],
    n: int = 50,
) -> list[QARecord]:
    """Return the first n QA records as the frozen debug subset.

    Always takes the first n records — never random.

    Args:
        qa_records: Full list of QA records from load_hotpotqa.
        n: Number of records to include. Default 50.

    Returns:
        List of the first n QARecord objects.

    Raises:
        ValueError: If n > len(qa_records).
    """
    if n > len(qa_records):
        raise ValueError(
            f"n={n} exceeds available records ({len(qa_records)})."
        )
    return qa_records[:n]


def save_records(
    qa_records: list[QARecord],
    corpus_documents: list[CorpusDocument],
    output_dir: Path,
) -> None:
    """Save processed QA records and corpus documents to disk as JSONL.

    Creates two files:
        {output_dir}/qa_records.jsonl  — one QARecord per line as JSON
        {output_dir}/corpus.jsonl      — one CorpusDocument per line as JSON

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


def load_records(
    output_dir: Path,
) -> tuple[list[QARecord], list[CorpusDocument]]:
    """Load previously saved QA records and corpus documents from JSONL files.

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
            d["supporting_facts"] = [
                SupportingFact(**sf) for sf in d["supporting_facts"]
            ]
            qa_records.append(QARecord(**d))

    corpus_documents: list[CorpusDocument] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            corpus_documents.append(CorpusDocument(**json.loads(line)))

    return qa_records, corpus_documents
