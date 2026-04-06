"""Tests for src/rag_evo/config/schema.py."""

import pytest

from rag_evo.config.schema import FullV1Config, MicroPilotConfig


def test_micro_pilot_default_valid() -> None:
    cfg = MicroPilotConfig()
    assert cfg.chunk_size == 256
    assert cfg.overlap == 0.0
    assert cfg.top_k == 5
    assert cfg.prompt_strategy == "zero-shot"
    assert cfg.temperature == 0.0


def test_micro_pilot_citations_required_always_true() -> None:
    cfg = MicroPilotConfig()
    assert cfg.citations_required is True
    with pytest.raises(TypeError):
        MicroPilotConfig(citations_required=False)


def test_micro_pilot_top_p_always_one() -> None:
    cfg = MicroPilotConfig()
    assert cfg.top_p == 1.0


def test_micro_pilot_to_dict_roundtrip() -> None:
    cfg = MicroPilotConfig(chunk_size=384, top_k=10, temperature=0.7)
    d = cfg.to_dict()
    assert isinstance(d, dict)
    cfg2 = MicroPilotConfig.from_dict(d)
    assert cfg.to_dict() == cfg2.to_dict()
    assert d["citations_required"] is True
    assert d["top_p"] == 1.0


def test_micro_pilot_to_json_roundtrip(tmp_path) -> None:
    cfg = MicroPilotConfig(chunk_size=512, overlap=0.2)
    path = tmp_path / "config.json"
    cfg.to_json(path)
    cfg2 = MicroPilotConfig.from_json(path)
    assert cfg.to_dict() == cfg2.to_dict()


def test_micro_pilot_copy_with() -> None:
    cfg = MicroPilotConfig(chunk_size=256, top_k=5)
    new_cfg = cfg.copy_with(chunk_size=512, top_k=10)
    assert new_cfg.chunk_size == 512
    assert new_cfg.top_k == 10
    assert new_cfg.overlap == cfg.overlap
    assert cfg.chunk_size == 256  # original unchanged


def test_full_v1_default_valid() -> None:
    cfg = FullV1Config()
    assert cfg.chunk_size == 256
    assert cfg.retriever_type == "BM25"
    assert cfg.reranker_enabled is False
    assert cfg.max_output_tokens == 256
    assert cfg.citations_required is True
    assert cfg.top_p == 1.0


def test_full_v1_has_all_eight_genes() -> None:
    cfg = FullV1Config()
    d = cfg.to_dict()
    expected_genes = {
        "chunk_size", "overlap", "top_k", "prompt_strategy",
        "temperature", "retriever_type", "reranker_enabled",
        "max_output_tokens",
    }
    assert expected_genes.issubset(d.keys())


def test_full_v1_to_dict_roundtrip() -> None:
    cfg = FullV1Config(
        retriever_type="hybrid",
        reranker_enabled=True,
        max_output_tokens=384,
    )
    d = cfg.to_dict()
    cfg2 = FullV1Config.from_dict(d)
    assert cfg.to_dict() == cfg2.to_dict()


def test_full_v1_to_json_roundtrip(tmp_path) -> None:
    cfg = FullV1Config(
        chunk_size=448,
        retriever_type="dense",
        max_output_tokens=128,
    )
    path = tmp_path / "full.json"
    cfg.to_json(path)
    cfg2 = FullV1Config.from_json(path)
    assert cfg.to_dict() == cfg2.to_dict()
