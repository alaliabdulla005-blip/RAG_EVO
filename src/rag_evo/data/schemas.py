"""Shared data schemas for the RAG-Evo pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SupportingFact:
    """A single supporting fact reference from HotpotQA.

    Args:
        doc_title: Title of the Wikipedia article containing this fact.
        sent_idx: Zero-based index of the supporting sentence within that article.
    """

    doc_title: str
    sent_idx: int


@dataclass(frozen=True)
class CorpusDocument:
    """A single document in the retrieval corpus.

    Args:
        doc_id: Unique identifier derived from the Wikipedia article title.
            Constructed as: title.strip().replace(" ", "_").lower()
        title: Original Wikipedia article title.
        sentences: List of sentences in this document, in order.
        text: Full document text formed by joining sentences with a space.
    """

    doc_id: str
    title: str
    sentences: list[str]
    text: str


@dataclass(frozen=True)
class QARecord:
    """A single processed question-answer record for the General QA track.

    Args:
        record_id: Unique identifier for this record. Use HotpotQA's _id field.
        question: The question string.
        gold_answer: The gold answer string.
        supporting_facts: List of SupportingFact references identifying the
            gold evidence sentences.
        source_doc_ids: List of doc_id strings for the documents containing
            gold evidence. Derived from supporting_facts doc_title fields
            using the same derivation rule as CorpusDocument.doc_id.
        split: The dataset split this record came from. One of: "train",
            "validation".
    """

    record_id: str
    question: str
    gold_answer: str
    supporting_facts: list[SupportingFact]
    source_doc_ids: list[str]
    split: str
