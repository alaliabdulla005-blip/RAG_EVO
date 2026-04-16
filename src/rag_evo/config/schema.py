"""Configuration schema for RAG-Evo genetic algorithm search space.

Defines the decision vector for micro-pilot (5 genes), full v1 (8 genes),
and full v2 (8 genes, Phase 16) chromosome versions. Every field corresponds
directly to a parameter described in the dissertation's Section 3.3.
"""

from __future__ import annotations

import hashlib
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

    @property
    def max_output_tokens(self) -> int:
        """Fixed max output tokens for MicroPilotConfig (not a search gene)."""
        return 150

    def canonical_hash(self) -> str:
        """SHA-256 hash of all output-affecting config fields."""
        return _canonical_hash(self)

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

    def canonical_hash(self) -> str:
        """SHA-256 hash of all output-affecting config fields."""
        return _canonical_hash(self)

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


@dataclass
class FullV2Config:
    """8-gene chromosome for Phase 16 GA experiments.

    Uses different valid ranges than FullV1Config:
      - chunk_size: {256, 384, 448, 512} (no 320)
      - overlap: [0.0, 0.5] (continuous)
      - prompt_strategy: adds "legal-citation"
      - retriever_type: {"BM25", "dense"} (no hybrid)
      - max_output_tokens: [64, 256] (continuous int)
    """

    chunk_size: int = 256
    overlap: float = 0.0
    top_k: int = 5
    prompt_strategy: str = "zero-shot"
    temperature: float = 0.0
    retriever_type: str = "BM25"
    reranker_enabled: bool = False
    max_output_tokens: int = 150

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
    def from_dict(cls, data: dict) -> FullV2Config:
        """Deserialise from a dictionary.

        Ignores fixed constants (citations_required, top_p) if present.
        """
        init_fields = {
            f.name for f in fields(cls) if f.init
        }
        filtered = {k: v for k, v in data.items() if k in init_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: Path) -> FullV2Config:
        """Load from a JSON file."""
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    def canonical_hash(self) -> str:
        """SHA-256 hash of all output-affecting config fields."""
        return _canonical_hash(self)

    def is_valid(self) -> bool:
        """Check whether this config is within v2 valid ranges."""
        from rag_evo.config.validators import is_valid_full_v2

        return is_valid_full_v2(self)

    def repair(self) -> FullV2Config:
        """Repair this config to enforce v2 validity."""
        from rag_evo.config.validators import repair_full_v2

        return repair_full_v2(self)

    def copy_with(self, **kwargs: object) -> FullV2Config:
        """Return a new FullV2Config with specified fields replaced."""
        d = self.to_dict()
        for k in _FIXED_CONSTANTS:
            d.pop(k, None)
        d.update(kwargs)
        return self.from_dict(d)


def _canonical_hash(
    config: MicroPilotConfig | FullV1Config | FullV2Config,
) -> str:
    """SHA-256 hash of all output-affecting config fields.

    MicroPilotConfig uses fixed defaults for the 3 extended genes
    so that its hash matches FullV2Config-at-defaults when the
    5 shared genes are identical.
    """
    key = {
        "chunk_size": config.chunk_size,
        "overlap": config.overlap,
        "top_k": config.top_k,
        "prompt_strategy": config.prompt_strategy,
        "temperature": config.temperature,
        "retriever_type": getattr(config, "retriever_type", "BM25"),
        "reranker_enabled": getattr(config, "reranker_enabled", False),
        "max_output_tokens": config.max_output_tokens,
        "citations_required": True,
        "top_p": 1.0,
    }
    canonical = json.dumps(key, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
