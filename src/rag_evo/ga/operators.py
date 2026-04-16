"""GA operators: tournament selection, uniform crossover, mutation.

All operators accept an explicit random.Random instance for
reproducibility. Never call random.random() directly.
"""

from __future__ import annotations

import random
from dataclasses import asdict

from rag_evo.config.schema import FullV1Config, FullV2Config, MicroPilotConfig
from rag_evo.config.space import (
    CHUNK_SIZE_VALUES,
    CHUNK_SIZE_VALUES_V2,
    DENSE_OVERLAP_VALUES_V2,
    MAX_OUTPUT_TOKENS_VALUES,
    OVERLAP_VALUES,
    PROMPT_STRATEGIES,
    PROMPT_STRATEGIES_V2,
    RETRIEVER_TYPES,
    TEMPERATURE_RANGE,
    TOP_K_RANGE,
    SearchSpace,
)
from rag_evo.config.validators import (
    repair_full_v1,
    repair_full_v2,
    repair_micro_pilot,
)


def _is_snapped(
    value: float, snap_values: list[float], tol: float = 1e-9
) -> bool:
    """Check if value is within tolerance of any snap value."""
    return any(abs(value - v) < tol for v in snap_values)


def tournament_select(
    population: list[MicroPilotConfig | FullV1Config | FullV2Config],
    fitnesses: list[float],
    tournament_size: int,
    rng: random.Random,
) -> MicroPilotConfig | FullV1Config | FullV2Config:
    """Select one individual via tournament selection.

    Samples tournament_size individuals uniformly without replacement,
    returns the one with the highest fitness.

    Args:
        population: Current generation's configs.
        fitnesses: Parallel list of fitness values.
        tournament_size: Number of individuals per tournament.
        rng: Seeded RNG.

    Returns:
        The winning individual's config.
    """
    indices = rng.sample(range(len(population)), tournament_size)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]


def uniform_crossover(
    parent1: MicroPilotConfig | FullV1Config | FullV2Config,
    parent2: MicroPilotConfig | FullV1Config | FullV2Config,
    rng: random.Random,
    search_space: SearchSpace,
) -> MicroPilotConfig | FullV1Config | FullV2Config:
    """Uniform crossover: per gene, randomly pick from one parent.

    Args:
        parent1: First parent config.
        parent2: Second parent config.
        rng: Seeded RNG.
        search_space: Defines which genes to cross.

    Returns:
        A child config with genes drawn from both parents.
    """
    gene_names = search_space.get_gene_names()
    d1 = asdict(parent1)
    d2 = asdict(parent2)
    child: dict = {}
    for gene in gene_names:
        child[gene] = d1[gene] if rng.random() < 0.5 else d2[gene]

    if isinstance(parent1, MicroPilotConfig):
        return MicroPilotConfig.from_dict(child)
    if isinstance(parent1, FullV2Config):
        # Snap overlap to pre-built index value for dense retrieval
        if child.get("retriever_type") == "dense":
            if not _is_snapped(child["overlap"], DENSE_OVERLAP_VALUES_V2):
                child["overlap"] = min(
                    DENSE_OVERLAP_VALUES_V2,
                    key=lambda v: abs(v - child["overlap"]),
                )
        return FullV2Config.from_dict(child)
    return FullV1Config.from_dict(child)


def mutate(
    config: MicroPilotConfig | FullV1Config | FullV2Config,
    rng: random.Random,
    mutation_rate: float,
    search_space: SearchSpace,
) -> MicroPilotConfig | FullV1Config | FullV2Config:
    """Per-gene mutation with type-aware operators.

    Each gene is independently mutated with probability mutation_rate.
    After mutation, the result is repaired to ensure validity.

    Mutation strategies by gene type:
    - Discrete set genes: resample uniformly from valid values
    - Continuous range genes: Gaussian perturbation clamped to bounds
    - Integer range genes: Gaussian perturbation, rounded and clamped
    - Boolean genes: flip

    Args:
        config: Config to mutate.
        rng: Seeded RNG.
        mutation_rate: Per-gene mutation probability.
        search_space: Defines gene names and valid ranges.

    Returns:
        A new (repaired) config. Input config is never modified.
    """
    gene_names = search_space.get_gene_names()
    d = asdict(config)

    is_v2 = isinstance(config, FullV2Config)

    for gene in gene_names:
        if rng.random() >= mutation_rate:
            continue

        if gene == "chunk_size":
            vals = CHUNK_SIZE_VALUES_V2 if is_v2 else CHUNK_SIZE_VALUES
            d[gene] = rng.choice(vals)
        elif gene == "overlap":
            if is_v2:
                if d.get("retriever_type") == "dense":
                    d[gene] = rng.choice(DENSE_OVERLAP_VALUES_V2)
                else:
                    d[gene] = max(
                        0.0, min(0.5, d[gene] + rng.gauss(0, 0.05))
                    )
            else:
                d[gene] = rng.choice(OVERLAP_VALUES)
        elif gene == "top_k":
            if is_v2:
                step = rng.choice([-2, -1, 1, 2])
                d[gene] = max(1, min(20, d[gene] + step))
            else:
                new_val = d[gene] + rng.gauss(0, 3)
                d[gene] = int(
                    max(
                        TOP_K_RANGE[0],
                        min(TOP_K_RANGE[1], round(new_val)),
                    )
                )
        elif gene == "prompt_strategy":
            strats = PROMPT_STRATEGIES_V2 if is_v2 else PROMPT_STRATEGIES
            d[gene] = rng.choice(strats)
        elif gene == "temperature":
            new_val = d[gene] + rng.gauss(0, 0.1)
            d[gene] = max(
                TEMPERATURE_RANGE[0],
                min(TEMPERATURE_RANGE[1], new_val),
            )
        elif gene == "retriever_type":
            if is_v2:
                d[gene] = (
                    "dense" if d[gene] == "BM25" else "BM25"
                )
            else:
                d[gene] = rng.choice(RETRIEVER_TYPES)
        elif gene == "reranker_enabled":
            d[gene] = not d[gene]
        elif gene == "max_output_tokens":
            if is_v2:
                step = rng.choice([-32, -16, 16, 32])
                d[gene] = max(64, min(256, d[gene] + step))
            else:
                d[gene] = rng.choice(MAX_OUTPUT_TOKENS_VALUES)

    # If v2 dense retrieval ended up with a non-indexed overlap
    # (e.g. retriever_type was mutated to "dense" but overlap wasn't),
    # snap overlap to the nearest pre-built index value.
    if is_v2 and d.get("retriever_type") == "dense":
        if not _is_snapped(d["overlap"], DENSE_OVERLAP_VALUES_V2):
            d["overlap"] = min(
                DENSE_OVERLAP_VALUES_V2,
                key=lambda v: abs(v - d["overlap"]),
            )

    if isinstance(config, MicroPilotConfig):
        return repair_micro_pilot(MicroPilotConfig.from_dict(d))
    if is_v2:
        return repair_full_v2(FullV2Config.from_dict(d))
    return repair_full_v1(FullV1Config.from_dict(d))
