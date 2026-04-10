"""Evaluation harness for RAG-Evo.

Computes all metrics for a set of PipelineResults matched against
their corresponding QARecords.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_evo.data.schemas import QARecord
from rag_evo.evaluation.metrics import (
    compute_citation_f1,
    compute_citation_precision,
    compute_citation_recall,
    compute_em,
    compute_f1,
    compute_retrieval_recall_at_k,
    extract_cited_doc_ids,
)
from rag_evo.evaluation.normaliser import normalise_answer
from rag_evo.pipeline.rag_pipeline import PipelineResult


@dataclass(frozen=True)
class RecordMetrics:
    """All metrics for a single QARecord evaluation.

    Args:
        record_id: ID of the evaluated QARecord.
        em: Exact match score.
        f1: Token-level F1.
        citation_precision: Citation precision.
        citation_recall: Citation recall.
        citation_f1: Citation F1.
        retrieval_recall_at_k: Retrieval recall at the top_k
            used during retrieval.
        input_tokens: From PipelineResult.
        output_tokens: From PipelineResult.
        latency_ms: From PipelineResult.
    """

    record_id: str
    em: float
    f1: float
    citation_precision: float
    citation_recall: float
    citation_f1: float
    retrieval_recall_at_k: float
    input_tokens: int
    output_tokens: int
    latency_ms: float


@dataclass(frozen=True)
class AggregateMetrics:
    """Aggregate metrics over a full evaluation set.

    All float fields are means over the corresponding RecordMetrics fields.

    Args:
        num_records: Number of records evaluated.
        mean_em: Mean exact match.
        mean_f1: Mean token F1.
        mean_citation_precision: Mean citation precision.
        mean_citation_recall: Mean citation recall.
        mean_citation_f1: Mean citation F1.
        mean_retrieval_recall_at_k: Mean retrieval recall@k.
        mean_input_tokens: Mean input tokens per query.
        mean_output_tokens: Mean output tokens per query.
        mean_latency_ms: Mean latency per query in ms.
        total_input_tokens: Sum of all input tokens.
        total_output_tokens: Sum of all output tokens.
    """

    num_records: int
    mean_em: float
    mean_f1: float
    mean_citation_precision: float
    mean_citation_recall: float
    mean_citation_f1: float
    mean_retrieval_recall_at_k: float
    mean_input_tokens: float
    mean_output_tokens: float
    mean_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int


class EvaluationHarness:
    """Evaluation harness matching PipelineResults to QARecords.

    Args:
        qa_records: List of QARecord objects used as ground truth.
            Indexed internally by record_id for O(1) lookup.
    """

    def __init__(self, qa_records: list[QARecord]) -> None:
        self._records_by_id = {r.record_id: r for r in qa_records}

    def evaluate_result(
        self,
        result: PipelineResult,
    ) -> RecordMetrics:
        """Evaluate a single PipelineResult against its QARecord.

        Args:
            result: A PipelineResult to evaluate.

        Returns:
            RecordMetrics with all fields populated.

        Raises:
            KeyError: If result.record_id is not found in qa_records.
        """
        record = self._records_by_id[result.record_id]

        norm_pred = normalise_answer(result.answer)
        norm_gold = normalise_answer(record.gold_answer)

        em = compute_em(norm_pred, norm_gold)
        f1 = compute_f1(norm_pred, norm_gold)

        cited_doc_ids = extract_cited_doc_ids(
            result.answer, result.retrieved_chunks
        )
        gold_doc_ids = record.source_doc_ids

        cit_p = compute_citation_precision(cited_doc_ids, gold_doc_ids)
        cit_r = compute_citation_recall(cited_doc_ids, gold_doc_ids)
        cit_f1 = compute_citation_f1(cited_doc_ids, gold_doc_ids)

        ret_recall = compute_retrieval_recall_at_k(
            result.retrieved_chunks, gold_doc_ids, result.top_k
        )

        return RecordMetrics(
            record_id=result.record_id,
            em=em,
            f1=f1,
            citation_precision=cit_p,
            citation_recall=cit_r,
            citation_f1=cit_f1,
            retrieval_recall_at_k=ret_recall,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
        )

    def evaluate_batch(
        self,
        results: list[PipelineResult],
    ) -> tuple[list[RecordMetrics], AggregateMetrics]:
        """Evaluate a list of PipelineResults.

        Args:
            results: List of PipelineResult objects.

        Returns:
            Tuple of (per_record_metrics, aggregate_metrics).
        """
        per_record = [self.evaluate_result(r) for r in results]
        n = len(per_record)

        if n == 0:
            return [], AggregateMetrics(
                num_records=0,
                mean_em=0.0,
                mean_f1=0.0,
                mean_citation_precision=0.0,
                mean_citation_recall=0.0,
                mean_citation_f1=0.0,
                mean_retrieval_recall_at_k=0.0,
                mean_input_tokens=0.0,
                mean_output_tokens=0.0,
                mean_latency_ms=0.0,
                total_input_tokens=0,
                total_output_tokens=0,
            )

        aggregate = AggregateMetrics(
            num_records=n,
            mean_em=sum(m.em for m in per_record) / n,
            mean_f1=sum(m.f1 for m in per_record) / n,
            mean_citation_precision=sum(
                m.citation_precision for m in per_record
            )
            / n,
            mean_citation_recall=sum(
                m.citation_recall for m in per_record
            )
            / n,
            mean_citation_f1=sum(m.citation_f1 for m in per_record) / n,
            mean_retrieval_recall_at_k=sum(
                m.retrieval_recall_at_k for m in per_record
            )
            / n,
            mean_input_tokens=sum(m.input_tokens for m in per_record) / n,
            mean_output_tokens=sum(
                m.output_tokens for m in per_record
            )
            / n,
            mean_latency_ms=sum(m.latency_ms for m in per_record) / n,
            total_input_tokens=sum(m.input_tokens for m in per_record),
            total_output_tokens=sum(m.output_tokens for m in per_record),
        )

        return per_record, aggregate

    def save_metrics(
        self,
        per_record: list[RecordMetrics],
        aggregate: AggregateMetrics,
        output_dir: Path,
    ) -> None:
        """Save metrics to disk.

        Creates two files in output_dir:
        - per_record_metrics.jsonl: one RecordMetrics per line
        - aggregate_metrics.json: AggregateMetrics as single JSON object

        Args:
            per_record: List of RecordMetrics objects.
            aggregate: AggregateMetrics object.
            output_dir: Directory to write files. Created if missing.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        with (output_dir / "per_record_metrics.jsonl").open(
            "w", encoding="utf-8"
        ) as f:
            for m in per_record:
                f.write(json.dumps(asdict(m)) + "\n")

        with (output_dir / "aggregate_metrics.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(asdict(aggregate), f, indent=2)
