"""Tests for src/rag_evo/config/validators.py."""

from rag_evo.config.schema import FullV1Config, MicroPilotConfig
from rag_evo.config.validators import (
    is_valid_full_v1,
    is_valid_micro_pilot,
    repair_full_v1,
    repair_micro_pilot,
    snap_to_nearest,
)


def test_is_valid_micro_pilot_default() -> None:
    assert is_valid_micro_pilot(MicroPilotConfig()) is True


def test_is_valid_micro_pilot_invalid_chunk_size() -> None:
    cfg = MicroPilotConfig(chunk_size=100)
    assert is_valid_micro_pilot(cfg) is False


def test_is_valid_micro_pilot_invalid_overlap() -> None:
    cfg = MicroPilotConfig(overlap=0.5)
    assert is_valid_micro_pilot(cfg) is False


def test_is_valid_micro_pilot_invalid_top_k_low() -> None:
    cfg = MicroPilotConfig(top_k=0)
    assert is_valid_micro_pilot(cfg) is False


def test_is_valid_micro_pilot_invalid_top_k_high() -> None:
    cfg = MicroPilotConfig(top_k=21)
    assert is_valid_micro_pilot(cfg) is False


def test_is_valid_micro_pilot_invalid_temperature() -> None:
    cfg_high = MicroPilotConfig(temperature=1.5)
    assert is_valid_micro_pilot(cfg_high) is False
    cfg_low = MicroPilotConfig(temperature=-0.1)
    assert is_valid_micro_pilot(cfg_low) is False


def test_repair_micro_pilot_chunk_size() -> None:
    repaired = repair_micro_pilot(MicroPilotConfig(chunk_size=300))
    assert repaired.chunk_size == 320
    repaired2 = repair_micro_pilot(MicroPilotConfig(chunk_size=100))
    assert repaired2.chunk_size == 256


def test_repair_micro_pilot_overlap() -> None:
    repaired = repair_micro_pilot(MicroPilotConfig(overlap=0.15))
    assert repaired.overlap == 0.1
    repaired2 = repair_micro_pilot(MicroPilotConfig(overlap=0.6))
    assert repaired2.overlap == 0.3


def test_repair_micro_pilot_top_k() -> None:
    repaired = repair_micro_pilot(MicroPilotConfig(top_k=0))
    assert repaired.top_k == 1
    repaired2 = repair_micro_pilot(MicroPilotConfig(top_k=25))
    assert repaired2.top_k == 20


def test_repair_micro_pilot_temperature() -> None:
    repaired = repair_micro_pilot(MicroPilotConfig(temperature=1.5))
    assert repaired.temperature == 1.0
    repaired2 = repair_micro_pilot(MicroPilotConfig(temperature=-0.5))
    assert repaired2.temperature == 0.0


def test_repair_micro_pilot_invalid_strategy() -> None:
    repaired = repair_micro_pilot(
        MicroPilotConfig(prompt_strategy="unknown")
    )
    assert repaired.prompt_strategy == "zero-shot"


def test_repair_preserves_valid_config() -> None:
    cfg = MicroPilotConfig(
        chunk_size=384, overlap=0.2, top_k=10,
        prompt_strategy="chain-of-thought", temperature=0.5,
    )
    repaired = repair_micro_pilot(cfg)
    assert repaired.to_dict() == cfg.to_dict()


def test_is_valid_full_v1_invalid_retriever() -> None:
    cfg = FullV1Config(retriever_type="random")
    assert is_valid_full_v1(cfg) is False


def test_repair_full_v1_invalid_retriever() -> None:
    repaired = repair_full_v1(FullV1Config(retriever_type="random"))
    assert repaired.retriever_type == "BM25"


def test_repair_full_v1_max_output_tokens() -> None:
    repaired = repair_full_v1(FullV1Config(max_output_tokens=100))
    assert repaired.max_output_tokens == 128
    repaired2 = repair_full_v1(FullV1Config(max_output_tokens=600))
    assert repaired2.max_output_tokens == 512


def test_snap_to_nearest_basic() -> None:
    assert snap_to_nearest(300, [256, 320, 384]) == 320
    assert snap_to_nearest(270, [256, 320, 384]) == 256
