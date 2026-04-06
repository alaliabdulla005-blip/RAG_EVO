"""Tests for src/rag_evo/cache/cache.py."""

from pathlib import Path

from rag_evo.cache.cache import ResultCache, make_cache_key
from rag_evo.config.schema import MicroPilotConfig
from rag_evo.data.schemas import QARecord, SupportingFact
from rag_evo.pipeline.rag_pipeline import PipelineResult
from rag_evo.retrieval.chunker import Chunk


def _record(record_id: str = "r001") -> QARecord:
    return QARecord(
        record_id=record_id,
        question="What is the answer?",
        gold_answer="42",
        supporting_facts=[SupportingFact(doc_title="Doc", sent_idx=0)],
        source_doc_ids=["doc0"],
        split="validation",
    )


def _config(**kwargs) -> MicroPilotConfig:
    return MicroPilotConfig(**kwargs)


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="doc0__chunk_0000",
        doc_id="doc0",
        text="test text",
        token_count=2,
        char_start=0,
        char_end=9,
    )


def _pipeline_result(record_id: str = "r001") -> PipelineResult:
    return PipelineResult(
        record_id=record_id,
        question="What is the answer?",
        answer="42",
        cited_chunk_ids=["[DOC_1]"],
        retrieved_chunks=[_chunk()],
        input_tokens=100,
        output_tokens=5,
        latency_ms=50.0,
        retriever_type="bm25",
        prompt_strategy="zero-shot",
        chunk_size=256,
        overlap=0.0,
        top_k=5,
        raw_prompt="Context:\n[DOC_1]: test text\n\nQuestion: What is the answer?",
        raw_answer="42",
    )


# ── make_cache_key ────────────────────────────────────────────────────────────

def test_make_cache_key_deterministic() -> None:
    r = _record()
    cfg = _config()
    key1 = make_cache_key(r, cfg, "model-a")
    key2 = make_cache_key(r, cfg, "model-a")
    assert key1 == key2
    assert len(key1) == 64


def test_make_cache_key_differs_by_record() -> None:
    cfg = _config()
    k1 = make_cache_key(_record("r001"), cfg, "m")
    k2 = make_cache_key(_record("r002"), cfg, "m")
    assert k1 != k2


def test_make_cache_key_differs_by_config() -> None:
    r = _record()
    k1 = make_cache_key(r, _config(chunk_size=256), "m")
    k2 = make_cache_key(r, _config(chunk_size=512), "m")
    assert k1 != k2


def test_make_cache_key_differs_by_model() -> None:
    r = _record()
    cfg = _config()
    k1 = make_cache_key(r, cfg, "model-a")
    k2 = make_cache_key(r, cfg, "model-b")
    assert k1 != k2


def test_make_cache_key_covers_all_genes() -> None:
    r = _record()
    base = make_cache_key(r, _config(), "m")
    assert base != make_cache_key(r, _config(chunk_size=512), "m")
    assert base != make_cache_key(r, _config(overlap=0.2), "m")
    assert base != make_cache_key(r, _config(top_k=10), "m")
    assert base != make_cache_key(r, _config(prompt_strategy="chain-of-thought"), "m")
    assert base != make_cache_key(r, _config(temperature=0.5), "m")
    assert base != make_cache_key(r, _config(), "model-x")


# ── ResultCache ───────────────────────────────────────────────────────────────

def test_cache_set_and_get(tmp_path: Path) -> None:
    cache = ResultCache(tmp_path / "cache")
    key = make_cache_key(_record(), _config(), "m")
    result = _pipeline_result()
    cache.set(key, result)
    loaded = cache.get(key)
    assert loaded is not None
    assert loaded.record_id == result.record_id
    assert loaded.answer == result.answer


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    cache = ResultCache(tmp_path / "cache")
    assert cache.get("nonexistent" * 4) is None


def test_cache_exists(tmp_path: Path) -> None:
    cache = ResultCache(tmp_path / "cache")
    key = make_cache_key(_record(), _config(), "m")
    assert cache.exists(key) is False
    cache.set(key, _pipeline_result())
    assert cache.exists(key) is True


def test_cache_stats_hit_rate(tmp_path: Path) -> None:
    cache = ResultCache(tmp_path / "cache")
    key = make_cache_key(_record(), _config(), "m")
    cache.set(key, _pipeline_result())

    cache.get(key)   # hit
    cache.get(key)   # hit
    cache.get("x" * 64)  # miss

    s = cache.stats()
    assert s["hits"] == 2
    assert s["misses"] == 1
    assert abs(s["hit_rate"] - 2 / 3) < 1e-9


def test_cache_disabled(tmp_path: Path) -> None:
    cache = ResultCache(tmp_path / "cache", enabled=False)
    key = make_cache_key(_record(), _config(), "m")
    cache.set(key, _pipeline_result())
    assert cache.get(key) is None


def test_cache_size(tmp_path: Path) -> None:
    cache = ResultCache(tmp_path / "cache")
    assert cache.size() == 0
    cache.set(make_cache_key(_record(), _config(), "m"), _pipeline_result())
    assert cache.size() == 1


def test_cache_roundtrip_preserves_pipeline_result(tmp_path: Path) -> None:
    cache = ResultCache(tmp_path / "cache")
    key = make_cache_key(_record(), _config(), "m")
    original = _pipeline_result()
    cache.set(key, original)
    loaded = cache.get(key)
    assert loaded is not None
    assert loaded.answer == original.answer
    assert loaded.record_id == original.record_id
    assert loaded.chunk_size == original.chunk_size
    assert loaded.retrieved_chunks[0].doc_id == original.retrieved_chunks[0].doc_id
