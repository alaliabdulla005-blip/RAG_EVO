"""Configuration schema for RAG-Evo genetic algorithm search space.

Defines the decision vector for both micro-pilot (5 genes) and
full v1 (8 genes) chromosome versions. Every field corresponds
directly to a parameter described in the dissertation's Section 3.3.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

_FIXED_CONSTANTS = {"citations_required", "top_p"}


@dataclass
class MicroPilotConfig:
    """5-gene chromosome for the micro-pilot GA experiment.

    This is the decision vector for Phase 10 pilot runs.
    All five parameters are actively searched by the GA.

    Fixed constants not in this schema:
        citations_required = True (always)
        top_p = 1.0 (always)

    Args:
        chunk_size: Token count per chunk. Discrete in [256, 512],
            step 64. Valid values: {256, 320, 384, 448, 512}.
        overlap: Fractional overlap between chunks.
            Discrete: {0.0, 0.1, 0.2, 0.3}.
        top_k: Number of chunks retrieved per query.
            Integer in [1, 20].
        prompt_strategy: Prompt template to use.
            One of {"zero-shot", "chain-of-thought"}.
        temperature: Sampling temperature.
            Float in [0.0, 1.0].
    """

    chunk_size: int = 256
    overlap: float = 0.0
    top_k: int = 5
    prompt_strategy: str = "zero-shot"
    temperature: float = 0.0

    # Fixed constants — always present, never searched
    citations_required: bool = field(default=True, init=False)
    top_p: float = field(default=1.0, init=False)

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return asdict(self)

    def to_json(self, path: Path) -> None:
        """Save to a JSON file at path. Creates parent dirs."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_dict(cls, data: dict) -> MicroPilotConfig:
        """Deserialise from a dictionary.

        Ignores fixed constants (citations_required, top_p) if present.
        """
        init_fields = {
            f.name for f in fields(cls) if f.init
        }
        filtered = {k: v for k, v in data.items() if k in init_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: Path) -> MicroPilotConfig:
        """Load from a JSON file."""
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    def copy_with(self, **kwargs: object) -> MicroPilotConfig:
        """Return a new MicroPilotConfig with specified fields replaced.

        Used by mutation operators to produce modified configs.
        Does not modify self.
        """
        d = self.to_dict()
        for k in _FIXED_CONSTANTS:
            d.pop(k, None)
        d.update(kwargs)
        return self.from_dict(d)


@dataclass
class FullV1Config:
    """8-gene chromosome for the full v1 GA experiment.

    Extends MicroPilotConfig with three additional retrieval genes.
    Used for Phase 12 full General-QA experiments and Phase 14
    legal track experiments.

    Additional args (beyond MicroPilotConfig):
        retriever_type: Retrieval method.
            One of {"BM25", "dense", "hybrid"}.
        reranker_enabled: Whether to apply cross-encoder reranking.
            Boolean gene.
        max_output_tokens: Maximum tokens to generate per query.
            Integer in [128, 512], step 64.
            Valid values: {128, 192, 256, 320, 384, 448, 512}.
    """

    chunk_size: int = 256
    overlap: float = 0.0
    top_k: int = 5
    prompt_strategy: str = "zero-shot"
    temperature: float = 0.0
    retriever_type: str = "BM25"
    reranker_enabled: bool = False
    max_output_tokens: int = 256

    # Fixed constants
    citations_required: bool = field(default=True, init=False)
    top_p: float = field(default=1.0, init=False)

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return asdict(self)

    def to_json(self, path: Path) -> None:
        """Save to a JSON file at path. Creates parent dirs."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_dict(cls, data: dict) -> FullV1Config:
        """Deserialise from a dictionary.

        Ignores fixed constants (citations_required, top_p) if present.
        """
        init_fields = {
            f.name for f in fields(cls) if f.init
        }
        filtered = {k: v for k, v in data.items() if k in init_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: Path) -> FullV1Config:
        """Load from a JSON file."""
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    def copy_with(self, **kwargs: object) -> FullV1Config:
        """Return a new FullV1Config with specified fields replaced.

        Used by mutation operators to produce modified configs.
        Does not modify self.
        """
        d = self.to_dict()
        for k in _FIXED_CONSTANTS:
            d.pop(k, None)
        d.update(kwargs)
        return self.from_dict(d)
