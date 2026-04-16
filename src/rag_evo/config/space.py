"""Search space definition for RAG-Evo genetic algorithm.

Defines valid ranges, discrete sets, and sampling logic for each gene.
The GA uses this module to initialise populations and apply mutations.
"""

from __future__ import annotations

import random
from typing import Any

from rag_evo.config.schema import FullV1Config, FullV2Config, MicroPilotConfig

# Micro-pilot gene spaces
CHUNK_SIZE_VALUES: list[int] = [256, 320, 384, 448, 512]
OVERLAP_VALUES: list[float] = [0.0, 0.1, 0.2, 0.3]
TOP_K_RANGE: tuple[int, int] = (1, 20)
PROMPT_STRATEGIES: list[str] = ["zero-shot", "chain-of-thought"]
TEMPERATURE_RANGE: tuple[float, float] = (0.0, 1.0)

# Full v1 additional gene spaces
RETRIEVER_TYPES: list[str] = ["BM25", "dense", "hybrid"]
MAX_OUTPUT_TOKENS_VALUES: list[int] = [128, 192, 256, 320, 384, 448, 512]

# Full v2 gene spaces (Phase 16)
CHUNK_SIZE_VALUES_V2: list[int] = [256, 384, 448, 512]
OVERLAP_RANGE_V2: tuple[float, float] = (0.0, 0.5)
PROMPT_STRATEGIES_V2: list[str] = [
    "zero-shot",
    "chain-of-thought",
    "legal-citation",
]
RETRIEVER_TYPES_V2: list[str] = ["BM25", "dense"]
MAX_OUTPUT_TOKENS_RANGE_V2: tuple[int, int] = (64, 256)
# Discrete overlap values that have pre-built dense indexes.
# Dense retrieval MUST use one of these — continuous values segfault
# on 8GB RAM because on-the-fly index building exhausts memory.
DENSE_OVERLAP_VALUES_V2: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

_VALID_VERSIONS = {"micro_pilot", "full_v1", "full_v2"}

_MICRO_GENE_NAMES: list[str] = [
    "chunk_size",
    "overlap",
    "top_k",
    "prompt_strategy",
    "temperature",
]

_FULL_GENE_NAMES: list[str] = [
    *_MICRO_GENE_NAMES,
    "retriever_type",
    "reranker_enabled",
    "max_output_tokens",
]

_FULL_V2_GENE_NAMES: list[str] = [
    *_MICRO_GENE_NAMES,
    "retriever_type",
    "reranker_enabled",
    "max_output_tokens",
]


class SearchSpace:
    """Defines the valid search space for one chromosome version.

    Args:
        version: One of "micro_pilot", "full_v1", or "full_v2".

    Raises:
        ValueError: If version is unknown.
    """

    def __init__(self, version: str) -> None:
        if version not in _VALID_VERSIONS:
            raise ValueError(
                f"version must be one of {_VALID_VERSIONS}, "
                f"got {version!r}"
            )
        self._version = version

    def sample_random(
        self,
        rng: random.Random,
    ) -> MicroPilotConfig | FullV1Config | FullV2Config:
        """Sample a uniformly random valid configuration.

        Args:
            rng: Random number generator. Must be seeded externally.
                Never call random.random() directly — always use rng.

        Returns:
            A valid configuration sampled from the search space.
        """
        if self._version == "full_v2":
            retriever_type = rng.choice(RETRIEVER_TYPES_V2)
            if retriever_type == "dense":
                overlap = rng.choice(DENSE_OVERLAP_VALUES_V2)
            else:
                overlap = rng.uniform(*OVERLAP_RANGE_V2)
            return FullV2Config(
                chunk_size=rng.choice(CHUNK_SIZE_VALUES_V2),
                overlap=overlap,
                top_k=rng.randint(*TOP_K_RANGE),
                prompt_strategy=rng.choice(PROMPT_STRATEGIES_V2),
                temperature=rng.uniform(*TEMPERATURE_RANGE),
                retriever_type=retriever_type,
                reranker_enabled=rng.choice([True, False]),
                max_output_tokens=rng.randint(
                    *MAX_OUTPUT_TOKENS_RANGE_V2
                ),
            )

        kwargs: dict[str, Any] = {
            "chunk_size": rng.choice(CHUNK_SIZE_VALUES),
            "overlap": rng.choice(OVERLAP_VALUES),
            "top_k": rng.randint(*TOP_K_RANGE),
            "prompt_strategy": rng.choice(PROMPT_STRATEGIES),
            "temperature": rng.uniform(*TEMPERATURE_RANGE),
        }

        if self._version == "micro_pilot":
            return MicroPilotConfig(**kwargs)

        kwargs["retriever_type"] = rng.choice(RETRIEVER_TYPES)
        kwargs["reranker_enabled"] = rng.choice([True, False])
        kwargs["max_output_tokens"] = rng.choice(MAX_OUTPUT_TOKENS_VALUES)
        return FullV1Config(**kwargs)

    def get_gene_names(self) -> list[str]:
        """Return list of searchable gene names for this version.

        Returns only the genes the GA searches — not fixed constants.
        """
        if self._version == "micro_pilot":
            return list(_MICRO_GENE_NAMES)
        if self._version == "full_v2":
            return list(_FULL_V2_GENE_NAMES)
        return list(_FULL_GENE_NAMES)

    def get_gene_bounds(self) -> dict[str, Any]:
        """Return bounds/valid values for each gene.

        Returns a dict mapping gene name to its valid range or set.
        Used for logging and dissertation reporting.
        """
        if self._version == "full_v2":
            return {
                "chunk_size": CHUNK_SIZE_VALUES_V2,
                "overlap": OVERLAP_RANGE_V2,
                "top_k": TOP_K_RANGE,
                "prompt_strategy": PROMPT_STRATEGIES_V2,
                "temperature": TEMPERATURE_RANGE,
                "retriever_type": RETRIEVER_TYPES_V2,
                "reranker_enabled": [True, False],
                "max_output_tokens": MAX_OUTPUT_TOKENS_RANGE_V2,
            }

        bounds: dict[str, Any] = {
            "chunk_size": CHUNK_SIZE_VALUES,
            "overlap": OVERLAP_VALUES,
            "top_k": TOP_K_RANGE,
            "prompt_strategy": PROMPT_STRATEGIES,
            "temperature": TEMPERATURE_RANGE,
        }
        if self._version == "full_v1":
            bounds["retriever_type"] = RETRIEVER_TYPES
            bounds["reranker_enabled"] = [True, False]
            bounds["max_output_tokens"] = MAX_OUTPUT_TOKENS_VALUES
        return bounds
