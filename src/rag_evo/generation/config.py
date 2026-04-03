"""Decoding configuration for generation calls."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodingConfig:
    """Decoding parameters for a single generation call.

    Args:
        temperature: Sampling temperature. Set to 0.0 for deterministic output.
        max_tokens: Maximum number of tokens to generate.
        seed: Random seed for reproducibility. Only effective when temperature=0.0.
            Uses GLOBAL_SEED from environment if not specified.
        top_p: Nucleus sampling parameter. Fixed to 1.0 throughout v1 — do not
            expose this as a variable parameter.
    """

    temperature: float = 0.0
    max_tokens: int = 512
    seed: int = 42
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError(
                f"temperature must be in [0.0, 1.0], got {self.temperature}"
            )
        if not 1 <= self.max_tokens <= 2048:
            raise ValueError(
                f"max_tokens must be in [1, 2048], got {self.max_tokens}"
            )
        if self.top_p != 1.0:
            raise ValueError("top_p is fixed to 1.0 in v1. Do not change it.")
