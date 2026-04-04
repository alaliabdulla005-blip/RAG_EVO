"""Minimal end-to-end RAG pipeline for RAG-Evo.

Connects BM25Retriever → PromptBuilder → GeneratorWrapper
into a single callable unit. This is the naive RAG baseline.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_evo.data.schemas import QARecord
from rag_evo.generation.config import DecodingConfig
from rag_evo.generation.wrapper import GeneratorWrapper
from rag_evo.pipeline.prompt_builder import build_prompt_components
from rag_evo.retrieval.bm25_retriever import BM25Retriever
from rag_evo.retrieval.chunker import Chunk

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[DOC_\d+\]")


@dataclass(frozen=True)
class PipelineResult:
    """Complete result for a single RAG pipeline run on one query.

    All fields are mandatory. This object is saved to disk and consumed
    by the evaluation harness in Phase 6.

    Args:
        record_id: ID of the QARecord this result corresponds to.
        question: The question string.
        answer: The generated answer.
        cited_chunk_ids: List of [DOC_N] labels cited in the answer.
        retrieved_chunks: The chunks passed to the generator, in rank order.
        input_tokens: From WrapperOutput.
        output_tokens: From WrapperOutput.
        latency_ms: From WrapperOutput.
        retriever_type: Always "bm25" for this phase.
        prompt_strategy: The prompt strategy used.
        chunk_size: The chunk_size used for this run.
        overlap: The overlap used for this run.
        top_k: The top_k used for this run.
        raw_prompt: From WrapperOutput.
        raw_answer: From WrapperOutput.
    """

    record_id: str
    question: str
    answer: str
    cited_chunk_ids: list[str]
    retrieved_chunks: list[Chunk]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    retriever_type: str
    prompt_strategy: str
    chunk_size: int
    overlap: float
    top_k: int
    raw_prompt: str
    raw_answer: str


class RAGPipeline:
    """Minimal end-to-end RAG pipeline.

    Connects retriever → prompt builder → generator.
    One instance per configuration. Stateless after construction.

    Args:
        retriever: A BM25Retriever instance already built over chunks.
        wrapper: A GeneratorWrapper instance.
        chunk_size: The chunk_size used to build the retriever's index.
        overlap: The overlap used to build the retriever's index.
        top_k: Number of chunks to retrieve per query. Must be in [1, 20].
        prompt_strategy: One of "zero-shot", "chain-of-thought".
        decoding: DecodingConfig to pass to the wrapper.
    """

    def __init__(
        self,
        retriever: BM25Retriever,
        wrapper: GeneratorWrapper,
        chunk_size: int,
        overlap: float,
        top_k: int,
        prompt_strategy: str = "zero-shot",
        decoding: DecodingConfig | None = None,
    ) -> None:
        self._retriever = retriever
        self._wrapper = wrapper
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._top_k = top_k
        self._prompt_strategy = prompt_strategy
        self._decoding = decoding

    def run(self, record: QARecord) -> PipelineResult:
        """Run the full RAG pipeline on a single QARecord.

        Args:
            record: The QARecord to process.

        Returns:
            PipelineResult with all fields populated.
        """
        retrieval = self._retriever.retrieve(record.question, top_k=self._top_k)
        chunks = [chunk for chunk, _ in retrieval.retrieved_chunks]

        system_prompt, context_block = build_prompt_components(
            chunks, record.question, self._prompt_strategy
        )

        output = self._wrapper.generate(
            system_prompt=system_prompt,
            context_block=context_block,
            query=record.question,
            decoding=self._decoding,
        )

        cited_chunk_ids = _CITATION_RE.findall(output.answer)

        return PipelineResult(
            record_id=record.record_id,
            question=record.question,
            answer=output.answer,
            cited_chunk_ids=cited_chunk_ids,
            retrieved_chunks=chunks,
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
            latency_ms=output.latency_ms,
            retriever_type=retrieval.retriever_type,
            prompt_strategy=self._prompt_strategy,
            chunk_size=self._chunk_size,
            overlap=self._overlap,
            top_k=self._top_k,
            raw_prompt=output.raw_prompt,
            raw_answer=output.raw_answer,
        )

    def run_batch(
        self,
        records: list[QARecord],
        log_every: int = 10,
    ) -> list[PipelineResult]:
        """Run the pipeline on a list of QARecords.

        Args:
            records: List of QARecord objects to process.
            log_every: Log progress every N records.

        Returns:
            List of PipelineResult objects. May be shorter than records
            if any records failed.
        """
        results: list[PipelineResult] = []
        for i, record in enumerate(records, start=1):
            try:
                results.append(self.run(record))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping record %s: %s", record.record_id, exc
                )
            if i % log_every == 0:
                logger.info("Processed %d / %d records.", i, len(records))
        return results


def save_results(results: list[PipelineResult], output_path: Path) -> None:
    """Save pipeline results to a JSONL file.

    Args:
        results: List of PipelineResult objects.
        output_path: Full path to the output .jsonl file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(asdict(result)) + "\n")


def load_results(input_path: Path) -> list[PipelineResult]:
    """Load pipeline results from a JSONL file.

    Args:
        input_path: Path to the .jsonl file.

    Returns:
        List of PipelineResult objects.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")
    results: list[PipelineResult] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            d["retrieved_chunks"] = [Chunk(**c) for c in d["retrieved_chunks"]]
            results.append(PipelineResult(**d))
    return results
