"""Validation and repair functions for RAG-Evo configurations.

repair() is called after every mutation to enforce validity.
is_valid() is called for logging and assertions only.
"""

from __future__ import annotations

from typing import Any

from rag_evo.config.schema import FullV1Config, MicroPilotConfig
from rag_evo.config.space import (
    CHUNK_SIZE_VALUES,
    MAX_OUTPUT_TOKENS_VALUES,
    OVERLAP_VALUES,
    PROMPT_STRATEGIES,
    RETRIEVER_TYPES,
    TEMPERATURE_RANGE,
    TOP_K_RANGE,
)


def snap_to_nearest(value: float, valid_values: list[Any]) -> Any:
    """Snap a value to the nearest element in a list of valid values.

    Args:
        value: The value to snap.
        valid_values: List of valid values.

    Returns:
        The element in valid_values closest to value.
    """
    return min(valid_values, key=lambda v: abs(v - value))


def is_valid_micro_pilot(config: MicroPilotConfig) -> bool:
    """Check whether a MicroPilotConfig is valid.

    Args:
        config: Configuration to check.

    Returns:
        True if all genes are within valid ranges, False otherwise.
    """
    if config.chunk_size not in CHUNK_SIZE_VALUES:
        return False
    if config.overlap not in OVERLAP_VALUES:
        return False
    if not (
        isinstance(config.top_k, int)
        and TOP_K_RANGE[0] <= config.top_k <= TOP_K_RANGE[1]
    ):
        return False
    if config.prompt_strategy not in PROMPT_STRATEGIES:
        return False
    if not TEMPERATURE_RANGE[0] <= config.temperature <= TEMPERATURE_RANGE[1]:
        return False
    if config.citations_required is not True:
        return False
    if config.top_p != 1.0:
        return False
    return True


def repair_micro_pilot(config: MicroPilotConfig) -> MicroPilotConfig:
    """Repair a MicroPilotConfig to enforce validity.

    Called after every mutation. Never raises — always returns
    a valid config. Clamps or snaps each gene to its nearest
    valid value.

    Args:
        config: Potentially invalid configuration after mutation.

    Returns:
        New MicroPilotConfig with all genes in valid ranges.
        Input config is never modified.
    """
    chunk_size = snap_to_nearest(config.chunk_size, CHUNK_SIZE_VALUES)
    overlap = snap_to_nearest(config.overlap, OVERLAP_VALUES)
    top_k = int(
        max(TOP_K_RANGE[0], min(TOP_K_RANGE[1], round(config.top_k)))
    )
    prompt_strategy = (
        config.prompt_strategy
        if config.prompt_strategy in PROMPT_STRATEGIES
        else "zero-shot"
    )
    temperature = max(
        TEMPERATURE_RANGE[0], min(TEMPERATURE_RANGE[1], config.temperature)
    )

    return MicroPilotConfig(
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k,
        prompt_strategy=prompt_strategy,
        temperature=temperature,
    )


def is_valid_full_v1(config: FullV1Config) -> bool:
    """Check whether a FullV1Config is valid.

    Args:
        config: Configuration to check.

    Returns:
        True if all genes are within valid ranges, False otherwise.
    """
    # Check the five shared genes via a temporary MicroPilotConfig
    micro = MicroPilotConfig(
        chunk_size=config.chunk_size,
        overlap=config.overlap,
        top_k=config.top_k,
        prompt_strategy=config.prompt_strategy,
        temperature=config.temperature,
    )
    if not is_valid_micro_pilot(micro):
        return False
    if config.retriever_type not in RETRIEVER_TYPES:
        return False
    if not isinstance(config.reranker_enabled, bool):
        return False
    if config.max_output_tokens not in MAX_OUTPUT_TOKENS_VALUES:
        return False
    if config.citations_required is not True:
        return False
    if config.top_p != 1.0:
        return False
    return True


def repair_full_v1(config: FullV1Config) -> FullV1Config:
    """Repair a FullV1Config to enforce validity.

    Applies all micro-pilot repairs plus additional gene repairs.

    Args:
        config: Potentially invalid configuration after mutation.

    Returns:
        New FullV1Config with all genes in valid ranges.
    """
    # Repair the shared micro-pilot genes
    repaired_micro = repair_micro_pilot(
        MicroPilotConfig(
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            top_k=config.top_k,
            prompt_strategy=config.prompt_strategy,
            temperature=config.temperature,
        )
    )

    retriever_type = (
        config.retriever_type
        if config.retriever_type in RETRIEVER_TYPES
        else "BM25"
    )
    reranker_enabled = bool(config.reranker_enabled)
    max_output_tokens = snap_to_nearest(
        config.max_output_tokens, MAX_OUTPUT_TOKENS_VALUES
    )

    return FullV1Config(
        chunk_size=repaired_micro.chunk_size,
        overlap=repaired_micro.overlap,
        top_k=repaired_micro.top_k,
        prompt_strategy=repaired_micro.prompt_strategy,
        temperature=repaired_micro.temperature,
        retriever_type=retriever_type,
        reranker_enabled=reranker_enabled,
        max_output_tokens=max_output_tokens,
    )
