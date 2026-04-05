"""Tests for src/rag_evo/evaluation/metrics.py"""

from rag_evo.evaluation.metrics import (
    compute_citation_f1,
    compute_citation_precision,
    compute_citation_recall,
    compute_em,
    compute_f1,
    compute_retrieval_recall_at_k,
    extract_cited_doc_ids,
)
from rag_evo.retrieval.chunker import Chunk


def _chunk(doc_id: str, idx: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}__chunk_{idx:04d}",
        doc_id=doc_id,
        text="placeholder",
        token_count=1,
        char_start=0,
        char_end=11,
    )


# --- EM ---


def test_em_exact_match() -> None:
    assert compute_em("yes", "yes") == 1.0


def test_em_no_match() -> None:
    assert compute_em("yes", "no") == 0.0


def test_em_empty_both() -> None:
    assert compute_em("", "") == 1.0


# --- F1 ---


def test_f1_perfect() -> None:
    assert compute_f1("new york city", "new york city") == 1.0


def test_f1_partial() -> None:
    score = compute_f1("new york", "new york city")
    assert 0.0 < score < 1.0


def test_f1_no_overlap() -> None:
    assert compute_f1("paris", "london") == 0.0


def test_f1_empty_both() -> None:
    assert compute_f1("", "") == 1.0


def test_f1_one_empty() -> None:
    assert compute_f1("", "london") == 0.0
    assert compute_f1("paris", "") == 0.0


# --- Citation extraction ---


def test_extract_cited_doc_ids_basic() -> None:
    chunks = [_chunk("doc_a"), _chunk("doc_b"), _chunk("doc_c")]
    result = extract_cited_doc_ids("Based on [DOC_2]", chunks)
    assert result == ["doc_b"]


def test_extract_cited_doc_ids_deduplicates() -> None:
    chunks = [_chunk("doc_a"), _chunk("doc_b")]
    result = extract_cited_doc_ids("[DOC_1] and [DOC_1]", chunks)
    assert result == ["doc_a"]


def test_extract_cited_doc_ids_out_of_range() -> None:
    chunks = [_chunk("doc_a"), _chunk("doc_b"), _chunk("doc_c")]
    result = extract_cited_doc_ids("[DOC_9]", chunks)
    assert result == []


# --- Citation precision ---


def test_citation_precision_perfect() -> None:
    assert compute_citation_precision(["a"], ["a"]) == 1.0


def test_citation_precision_empty_cited() -> None:
    assert compute_citation_precision([], ["a"]) == 0.0


# --- Citation recall ---


def test_citation_recall_perfect() -> None:
    assert compute_citation_recall(["a", "b"], ["a", "b"]) == 1.0


def test_citation_recall_empty_gold() -> None:
    assert compute_citation_recall(["a"], []) == 1.0


# --- Citation F1 ---


def test_citation_f1_perfect() -> None:
    assert compute_citation_f1(["a"], ["a"]) == 1.0


def test_citation_f1_zero() -> None:
    assert compute_citation_f1(["a"], ["b"]) == 0.0


# --- Retrieval recall@k ---


def test_retrieval_recall_at_k_hit() -> None:
    chunks = [_chunk("doc_a"), _chunk("doc_b")]
    assert compute_retrieval_recall_at_k(chunks, ["doc_a"], k=2) > 0.0


def test_retrieval_recall_at_k_miss() -> None:
    chunks = [_chunk("doc_a"), _chunk("doc_b")]
    assert compute_retrieval_recall_at_k(chunks, ["doc_z"], k=2) == 0.0
