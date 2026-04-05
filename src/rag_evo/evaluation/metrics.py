"""Individual metric computation functions for RAG-Evo evaluation.

All functions take normalised strings or pre-processed inputs.
All metrics return float values in [0.0, 1.0].
"""

import re
from collections import Counter

from rag_evo.evaluation.normaliser import tokenise

_CITATION_RE = re.compile(r"\[DOC_(\d+)\]")


def compute_em(predicted: str, gold: str) -> float:
    """Compute Exact Match between predicted and gold answers.

    Both inputs must already be normalised by normalise_answer().

    Args:
        predicted: Normalised predicted answer.
        gold: Normalised gold answer.

    Returns:
        1.0 if strings are identical, 0.0 otherwise.
    """
    return 1.0 if predicted == gold else 0.0


def compute_f1(predicted: str, gold: str) -> float:
    """Compute token-level F1 between predicted and gold answers.

    Both inputs must already be normalised by normalise_answer().
    Uses bag-of-words token overlap (Counter intersection).

    Args:
        predicted: Normalised predicted answer.
        gold: Normalised gold answer.

    Returns:
        F1 score in [0.0, 1.0].
    """
    pred_tokens = tokenise(predicted)
    gold_tokens = tokenise(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_shared = sum(common.values())

    if num_shared == 0:
        return 0.0

    precision = num_shared / len(pred_tokens)
    recall = num_shared / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def extract_cited_doc_ids(
    answer: str,
    retrieved_chunks: list,
) -> list[str]:
    """Extract cited document IDs from an answer string.

    Maps [DOC_N] labels to doc_ids via retrieved_chunks.
    [DOC_1] -> retrieved_chunks[0].doc_id, etc.

    Args:
        answer: Raw answer string (not normalised).
        retrieved_chunks: List of Chunk objects in retrieval rank order.

    Returns:
        List of unique doc_id strings in order of first citation.
    """
    seen: dict[str, None] = {}
    for match in _CITATION_RE.finditer(answer):
        idx = int(match.group(1)) - 1  # 1-based to 0-based
        if 0 <= idx < len(retrieved_chunks):
            doc_id = retrieved_chunks[idx].doc_id
            if doc_id not in seen:
                seen[doc_id] = None
    return list(seen.keys())


def compute_citation_precision(
    cited_doc_ids: list[str],
    gold_doc_ids: list[str],
) -> float:
    """Compute citation precision.

    Precision = |cited intersection gold| / |cited|

    Args:
        cited_doc_ids: Doc IDs cited in the answer.
        gold_doc_ids: Gold doc IDs from QARecord.source_doc_ids.

    Returns:
        Precision in [0.0, 1.0].
    """
    if not cited_doc_ids:
        return 0.0
    if not gold_doc_ids:
        return 1.0
    gold_set = set(gold_doc_ids)
    hits = sum(1 for d in cited_doc_ids if d in gold_set)
    return hits / len(cited_doc_ids)


def compute_citation_recall(
    cited_doc_ids: list[str],
    gold_doc_ids: list[str],
) -> float:
    """Compute citation recall.

    Recall = |cited intersection gold| / |gold|

    Args:
        cited_doc_ids: Doc IDs cited in the answer.
        gold_doc_ids: Gold doc IDs from QARecord.source_doc_ids.

    Returns:
        Recall in [0.0, 1.0].
    """
    if not gold_doc_ids:
        return 1.0
    cited_set = set(cited_doc_ids)
    hits = sum(1 for d in gold_doc_ids if d in cited_set)
    return hits / len(gold_doc_ids)


def compute_citation_f1(
    cited_doc_ids: list[str],
    gold_doc_ids: list[str],
) -> float:
    """Compute citation F1 as harmonic mean of precision and recall.

    Args:
        cited_doc_ids: Doc IDs cited in the answer.
        gold_doc_ids: Gold doc IDs from QARecord.source_doc_ids.

    Returns:
        F1 in [0.0, 1.0].
    """
    p = compute_citation_precision(cited_doc_ids, gold_doc_ids)
    r = compute_citation_recall(cited_doc_ids, gold_doc_ids)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def compute_retrieval_recall_at_k(
    retrieved_chunks: list,
    gold_doc_ids: list[str],
    k: int,
) -> float:
    """Compute retrieval recall at k.

    Fraction of gold doc IDs in the top-k retrieved chunks.

    Args:
        retrieved_chunks: List of Chunk objects from PipelineResult.
        gold_doc_ids: Gold doc IDs from QARecord.source_doc_ids.
        k: Number of retrieved chunks to consider.

    Returns:
        Recall@k in [0.0, 1.0].
    """
    if not gold_doc_ids:
        return 1.0
    if not retrieved_chunks:
        return 0.0
    k = min(k, len(retrieved_chunks))
    retrieved_doc_ids = {c.doc_id for c in retrieved_chunks[:k]}
    hits = sum(1 for d in gold_doc_ids if d in retrieved_doc_ids)
    return hits / len(gold_doc_ids)
