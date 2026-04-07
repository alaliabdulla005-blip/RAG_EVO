"""Tests for src/rag_evo/ga/operators.py."""

import random

from rag_evo.config.schema import FullV1Config, MicroPilotConfig
from rag_evo.config.space import (
    CHUNK_SIZE_VALUES,
    OVERLAP_VALUES,
    PROMPT_STRATEGIES,
    TEMPERATURE_RANGE,
    TOP_K_RANGE,
    SearchSpace,
)
from rag_evo.config.validators import is_valid_full_v1, is_valid_micro_pilot
from rag_evo.ga.operators import mutate, tournament_select, uniform_crossover


def _micro_space() -> SearchSpace:
    return SearchSpace("micro_pilot")


def _full_space() -> SearchSpace:
    return SearchSpace("full_v1")


def _micro_pop() -> list[MicroPilotConfig]:
    return [
        MicroPilotConfig(chunk_size=256, temperature=0.1),
        MicroPilotConfig(chunk_size=320, temperature=0.3),
        MicroPilotConfig(chunk_size=384, temperature=0.5),
        MicroPilotConfig(chunk_size=448, temperature=0.7),
        MicroPilotConfig(chunk_size=512, temperature=0.9),
    ]


# ── tournament_select ────────────────────────────────────────────────────────


def test_tournament_select_picks_best() -> None:
    pop = _micro_pop()
    fitnesses = [0.1, 0.5, 0.9, 0.3, 0.2]
    # With tournament_size == len(pop), must pick index 2 (fitness=0.9)
    rng = random.Random(42)
    winner = tournament_select(pop, fitnesses, len(pop), rng)
    assert winner is pop[2]


def test_tournament_select_respects_rng_seed() -> None:
    pop = _micro_pop()
    fitnesses = [0.1, 0.5, 0.9, 0.3, 0.2]

    rng1 = random.Random(123)
    w1 = tournament_select(pop, fitnesses, 2, rng1)
    rng2 = random.Random(123)
    w2 = tournament_select(pop, fitnesses, 2, rng2)
    assert w1 is w2


# ── uniform_crossover ────────────────────────────────────────────────────────


def test_crossover_returns_valid_micro() -> None:
    p1 = MicroPilotConfig(chunk_size=256, overlap=0.0, top_k=3,
                          prompt_strategy="zero-shot", temperature=0.1)
    p2 = MicroPilotConfig(chunk_size=512, overlap=0.3, top_k=15,
                          prompt_strategy="chain-of-thought", temperature=0.9)
    rng = random.Random(42)
    child = uniform_crossover(p1, p2, rng, _micro_space())
    assert isinstance(child, MicroPilotConfig)
    assert is_valid_micro_pilot(child)


def test_crossover_returns_valid_full_v1() -> None:
    p1 = FullV1Config(chunk_size=256, retriever_type="BM25")
    p2 = FullV1Config(chunk_size=512, retriever_type="dense")
    rng = random.Random(42)
    child = uniform_crossover(p1, p2, rng, _full_space())
    assert isinstance(child, FullV1Config)
    assert is_valid_full_v1(child)


def test_crossover_genes_from_parents() -> None:
    """Every gene in the child must come from one parent."""
    p1 = MicroPilotConfig(chunk_size=256, overlap=0.0, top_k=1,
                          prompt_strategy="zero-shot", temperature=0.0)
    p2 = MicroPilotConfig(chunk_size=512, overlap=0.3, top_k=20,
                          prompt_strategy="chain-of-thought", temperature=1.0)
    rng = random.Random(0)
    child = uniform_crossover(p1, p2, rng, _micro_space())
    assert child.chunk_size in (256, 512)
    assert child.overlap in (0.0, 0.3)
    assert child.top_k in (1, 20)
    assert child.prompt_strategy in ("zero-shot", "chain-of-thought")
    assert child.temperature in (0.0, 1.0)


# ── mutate ────────────────────────────────────────────────────────────────────


def test_mutate_returns_valid_micro() -> None:
    config = MicroPilotConfig()
    rng = random.Random(42)
    # mutation_rate=1.0 to force all genes to mutate
    result = mutate(config, rng, 1.0, _micro_space())
    assert isinstance(result, MicroPilotConfig)
    assert is_valid_micro_pilot(result)


def test_mutate_returns_valid_full_v1() -> None:
    config = FullV1Config()
    rng = random.Random(42)
    result = mutate(config, rng, 1.0, _full_space())
    assert isinstance(result, FullV1Config)
    assert is_valid_full_v1(result)


def test_mutate_no_op_at_zero_rate() -> None:
    """With mutation_rate=0, output should match input."""
    config = MicroPilotConfig(
        chunk_size=384, overlap=0.2, top_k=10,
        prompt_strategy="chain-of-thought", temperature=0.5,
    )
    rng = random.Random(42)
    result = mutate(config, rng, 0.0, _micro_space())
    assert result.chunk_size == config.chunk_size
    assert result.overlap == config.overlap
    assert result.top_k == config.top_k
    assert result.prompt_strategy == config.prompt_strategy
    assert result.temperature == config.temperature


def test_mutate_deterministic_with_seed() -> None:
    config = MicroPilotConfig()
    r1 = mutate(config, random.Random(99), 0.5, _micro_space())
    r2 = mutate(config, random.Random(99), 0.5, _micro_space())
    assert r1 == r2


def test_mutate_respects_gene_bounds() -> None:
    """Even with extreme mutation, genes stay within bounds."""
    config = MicroPilotConfig(top_k=1, temperature=0.0)
    rng = random.Random(42)
    for _ in range(50):
        config = mutate(config, rng, 1.0, _micro_space())
        assert config.chunk_size in CHUNK_SIZE_VALUES
        assert config.overlap in OVERLAP_VALUES
        assert TOP_K_RANGE[0] <= config.top_k <= TOP_K_RANGE[1]
        assert config.prompt_strategy in PROMPT_STRATEGIES
        assert TEMPERATURE_RANGE[0] <= config.temperature <= TEMPERATURE_RANGE[1]
