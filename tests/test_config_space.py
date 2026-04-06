"""Tests for src/rag_evo/config/space.py."""

import random

import pytest

from rag_evo.config.space import SearchSpace
from rag_evo.config.validators import is_valid_full_v1, is_valid_micro_pilot


def test_search_space_micro_pilot_valid_version() -> None:
    SearchSpace("micro_pilot")


def test_search_space_full_v1_valid_version() -> None:
    SearchSpace("full_v1")


def test_search_space_invalid_version() -> None:
    with pytest.raises(ValueError):
        SearchSpace("unknown")


def test_sample_random_micro_pilot_returns_valid() -> None:
    space = SearchSpace("micro_pilot")
    rng = random.Random(42)
    for _ in range(100):
        cfg = space.sample_random(rng)
        assert is_valid_micro_pilot(cfg)


def test_sample_random_full_v1_returns_valid() -> None:
    space = SearchSpace("full_v1")
    rng = random.Random(42)
    for _ in range(100):
        cfg = space.sample_random(rng)
        assert is_valid_full_v1(cfg)


def test_sample_random_uses_rng() -> None:
    space = SearchSpace("micro_pilot")
    rng1 = random.Random(99)
    rng2 = random.Random(99)
    cfg1 = space.sample_random(rng1)
    cfg2 = space.sample_random(rng2)
    assert cfg1.to_dict() == cfg2.to_dict()


def test_get_gene_names_micro_pilot() -> None:
    space = SearchSpace("micro_pilot")
    names = space.get_gene_names()
    assert names == [
        "chunk_size", "overlap", "top_k",
        "prompt_strategy", "temperature",
    ]
    assert "citations_required" not in names
    assert "top_p" not in names


def test_get_gene_names_full_v1() -> None:
    space = SearchSpace("full_v1")
    names = space.get_gene_names()
    assert len(names) == 8
    assert "retriever_type" in names
    assert "reranker_enabled" in names
    assert "max_output_tokens" in names
    assert "citations_required" not in names
    assert "top_p" not in names
