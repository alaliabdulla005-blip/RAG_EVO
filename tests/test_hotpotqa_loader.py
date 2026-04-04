"""Tests for src/rag_evo/data/hotpotqa_loader.py — no real downloads."""

from pathlib import Path
from unittest.mock import patch

import pytest

from rag_evo.data.hotpotqa_loader import (
    build_debug_subset,
    load_hotpotqa,
    load_records,
    make_doc_id,
    save_records,
)
from rag_evo.data.schemas import CorpusDocument, QARecord, SupportingFact

# ---------------------------------------------------------------------------
# Fake dataset helpers
# ---------------------------------------------------------------------------


def _fake_example(
    eid: str,
    question: str,
    answer: str,
    sf_titles: list[str],
    sf_sent_ids: list[int],
    ctx_titles: list[str],
    ctx_sentences: list[list[str]],
) -> dict:
    return {
        "id": eid,
        "question": question,
        "answer": answer,
        "supporting_facts": {"title": sf_titles, "sent_id": sf_sent_ids},
        "context": {"title": ctx_titles, "sentences": ctx_sentences},
    }


def _two_examples() -> list[dict]:
    return [
        _fake_example(
            "e1", "Q1?", "A1",
            sf_titles=["Doc A"],
            sf_sent_ids=[0],
            ctx_titles=["Doc A", "Doc B"],
            ctx_sentences=[["Sent A1.", "Sent A2."], ["Sent B1."]],
        ),
        _fake_example(
            "e2", "Q2?", "A2",
            sf_titles=["Doc C"],
            sf_sent_ids=[1],
            ctx_titles=["Doc A", "Doc C"],
            ctx_sentences=[["Sent A1.", "Sent A2."], ["Sent C1.", "Sent C2."]],
        ),
    ]


def _five_examples() -> list[dict]:
    return [
        _fake_example(
            f"e{i}", f"Q{i}?", f"A{i}",
            sf_titles=[f"Doc {i}"],
            sf_sent_ids=[0],
            ctx_titles=[f"Doc {i}", f"Doc {i+10}"],
            ctx_sentences=[[f"S{i}a.", f"S{i}b."], [f"S{i+10}a."]],
        )
        for i in range(5)
    ]


def _fake_qa_record(record_id: str = "r1") -> QARecord:
    return QARecord(
        record_id=record_id,
        question="Q?",
        gold_answer="A",
        supporting_facts=[SupportingFact(doc_title="Doc X", sent_idx=0)],
        source_doc_ids=[make_doc_id("Doc X")],
        split="validation",
    )


def _fake_corpus_doc(title: str = "Doc X") -> CorpusDocument:
    return CorpusDocument(
        doc_id=make_doc_id(title),
        title=title,
        sentences=["Sentence one."],
        text="Sentence one.",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_hotpotqa_returns_correct_types() -> None:
    with patch(
        "rag_evo.data.hotpotqa_loader.load_dataset",
        return_value=_two_examples(),
    ):
        qa, corpus = load_hotpotqa(split="validation", max_records=2)

    assert isinstance(qa, list)
    assert isinstance(corpus, list)
    assert all(isinstance(r, QARecord) for r in qa)
    assert all(isinstance(d, CorpusDocument) for d in corpus)


def test_load_hotpotqa_deduplicates_corpus() -> None:
    # Both examples reference "Doc A" in their context — should appear once.
    with patch(
        "rag_evo.data.hotpotqa_loader.load_dataset",
        return_value=_two_examples(),
    ):
        _, corpus = load_hotpotqa(split="validation")

    doc_ids = [d.doc_id for d in corpus]
    assert doc_ids.count(make_doc_id("Doc A")) == 1


def test_load_hotpotqa_max_records_slices_qa_not_corpus() -> None:
    with patch(
        "rag_evo.data.hotpotqa_loader.load_dataset",
        return_value=_five_examples(),
    ):
        qa, corpus = load_hotpotqa(split="validation", max_records=2)

    assert len(qa) == 2
    # Corpus must contain docs from all 5 examples (10 unique docs)
    assert len(corpus) == 10


def test_load_hotpotqa_invalid_split_raises() -> None:
    with pytest.raises(ValueError):
        load_hotpotqa(split="test")


def test_build_debug_subset_returns_first_n() -> None:
    records = [_fake_qa_record(str(i)) for i in range(10)]
    subset = build_debug_subset(records, n=3)
    assert len(subset) == 3
    assert subset == records[:3]


def test_build_debug_subset_raises_if_n_too_large() -> None:
    records = [_fake_qa_record(str(i)) for i in range(5)]
    with pytest.raises(ValueError):
        build_debug_subset(records, n=10)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    qa = [_fake_qa_record("r1"), _fake_qa_record("r2")]
    corpus = [_fake_corpus_doc("Doc X"), _fake_corpus_doc("Doc Y")]

    save_records(qa, corpus, tmp_path)
    qa2, corpus2 = load_records(tmp_path)

    assert qa2 == qa
    assert corpus2 == corpus
