"""Result cache for RAG-Evo experiment runner.

Caches PipelineResult objects keyed by a hash of the full
configuration and query. Prevents redundant API calls during
GA evaluation when the same config is re-evaluated.

Cache key design: the key must cover every dimension that
affects the output. Missing a dimension causes silent result
corruption — the cache returns a stale result for a different
configuration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path

from rag_evo.config.schema import FullV1Config, FullV2Config, MicroPilotConfig
from rag_evo.data.schemas import QARecord
from rag_evo.pipeline.rag_pipeline import PipelineResult
from rag_evo.retrieval.chunker import Chunk

logger = logging.getLogger(__name__)


def make_cache_key(
    record: QARecord,
    config: MicroPilotConfig | FullV1Config | FullV2Config,
    model_id: str,
) -> str:
    """Compute a SHA-256 cache key for a (record, config) pair.

    The key covers every dimension that affects pipeline output.
    Changing any field in config or the record_id produces a
    different key.

    Args:
        record: The QARecord being evaluated.
        config: MicroPilotConfig or FullV1Config.
        model_id: Model identifier string (e.g. "llama-3.1-8b-instant").

    Returns:
        64-character hex string (SHA-256 hash).
    """
    key_dict = {
        "record_id": record.record_id,
        "chunk_size": config.chunk_size,
        "overlap": config.overlap,
        "top_k": config.top_k,
        "prompt_strategy": config.prompt_strategy,
        "temperature": config.temperature,
        "retriever_type": getattr(config, "retriever_type", "BM25"),
        "reranker_enabled": getattr(config, "reranker_enabled", False),
        "max_output_tokens": getattr(config, "max_output_tokens", 256),
        "citations_required": config.citations_required,
        "top_p": config.top_p,
        "model_id": model_id,
    }
    canonical = json.dumps(key_dict, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


class ResultCache:
    """Disk-backed cache for PipelineResult objects.

    Stores results as JSON files in a cache directory.
    One file per (record, config) pair, named by cache key.
    Thread-safe for single-process use.

    Args:
        cache_dir: Directory to store cached results.
            Created if it does not exist.
        enabled: If False, all get() calls return None and
            set() calls are no-ops. Useful for ablation runs
            where you want fresh results.
    """

    def __init__(self, cache_dir: Path, enabled: bool = True) -> None:
        self._cache_dir = Path(cache_dir)
        self._enabled = enabled
        self._hits = 0
        self._misses = 0
        # Protects _hits/_misses counters when multiple threads call
        # get()/set() concurrently (parallel GA individual evaluation).
        # File operations are safe without locking because every cache key
        # maps to a unique path (SHA-256 of record+config), so threads
        # always write to disjoint files.
        self._lock = threading.Lock()
        if self._enabled:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        return self._cache_dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> PipelineResult | None:
        """Retrieve a cached PipelineResult by key.

        Args:
            key: Cache key from make_cache_key().

        Returns:
            PipelineResult if cached, None if not found.
        """
        if not self._enabled:
            with self._lock:
                self._misses += 1
            return None

        path = self._path_for(key)
        if not path.exists():
            with self._lock:
                self._misses += 1
            logger.debug("Cache miss: %s...", key[:8])
            return None

        with self._lock:
            self._hits += 1
        logger.debug("Cache hit: %s...", key[:8])
        d = json.loads(path.read_text(encoding="utf-8"))
        d["retrieved_chunks"] = [Chunk(**c) for c in d["retrieved_chunks"]]
        return PipelineResult(**d)

    def set(self, key: str, result: PipelineResult) -> None:
        """Store a PipelineResult in the cache.

        Args:
            key: Cache key from make_cache_key().
            result: PipelineResult to cache.
        """
        if not self._enabled:
            return
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(result)) + "\n", encoding="utf-8"
        )

    def exists(self, key: str) -> bool:
        """Check whether a key exists in the cache."""
        if not self._enabled:
            return False
        return self._path_for(key).exists()

    def size(self) -> int:
        """Return the number of cached entries."""
        if not self._enabled or not self._cache_dir.exists():
            return 0
        return sum(1 for _ in self._cache_dir.rglob("*.json"))

    def clear(self) -> None:
        """Delete all cached entries. Irreversible."""
        if not self._enabled:
            return
        for f in self._cache_dir.rglob("*.json"):
            f.unlink()

    def stats(self) -> dict:
        """Return cache statistics.

        Returns dict with:
            hits: Number of successful get() calls this session.
            misses: Number of get() calls returning None.
            size: Current number of cached entries on disk.
            hit_rate: hits / (hits + misses) if any calls made, else 0.0.
        """
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": self.size(),
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }
