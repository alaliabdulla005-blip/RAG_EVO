"""Tests for src/rag_evo/data/schemas.py"""

from dataclasses import FrozenInstanceError

from rag_evo.data.hotpotqa_loader import make_doc_id
from rag_evo.data.schemas import CorpusDocument, QARecord, SupportingFact


def _make_qa_record(record_id: str = "r1") -> QARecord:
    return QARecord(
        record_id=record_id,
        question="Q?",
        gold_answer="A",
        supporting_facts=[SupportingFact(doc_title="Some Title", sent_idx=0)],
        source_doc_ids=[make_doc_id("Some Title")],
        split="validation",
    )


def _make_corpus_doc(title: str = "Some Title") -> CorpusDocument:
    return CorpusDocument(
        doc_id=make_doc_id(title),
        title=title,
        sentences=["Sentence one.", "Sentence two."],
        text="Sentence one. Sentence two.",
    )


def test_qa_record_is_frozen() -> None:
    record = _make_qa_record()
    try:
        record.question = "changed"  # type: ignore[misc]
        raise AssertionError("Expected FrozenInstanceError")
    except FrozenInstanceError:
        pass


def test_corpus_document_is_frozen() -> None:
    doc = _make_corpus_doc()
    try:
        doc.title = "changed"  # type: ignore[misc]
        raise AssertionError("Expected FrozenInstanceError")
    except FrozenInstanceError:
        pass


def test_make_doc_id_basic() -> None:
    assert make_doc_id("Eiffel Tower") == "eiffel_tower"


def test_make_doc_id_strips_whitespace() -> None:
    assert make_doc_id("  Barack Obama  ") == "barack_obama"


def test_make_doc_id_lowercase() -> None:
    assert make_doc_id("UPPER CASE") == "upper_case"


def test_source_doc_ids_derived_consistently() -> None:
    title = "Albert Einstein"
    record = QARecord(
        record_id="r1",
        question="Q?",
        gold_answer="A",
        supporting_facts=[SupportingFact(doc_title=title, sent_idx=0)],
        source_doc_ids=[make_doc_id(title)],
        split="validation",
    )
    doc = CorpusDocument(
        doc_id=make_doc_id(title),
        title=title,
        sentences=["He was a physicist."],
        text="He was a physicist.",
    )
    assert record.source_doc_ids[0] == doc.doc_id
