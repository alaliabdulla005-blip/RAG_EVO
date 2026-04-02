"""Tests for src/rag_evo/utils/seed.py"""

from rag_evo.utils.seed import set_global_seed


def test_set_global_seed_runs_without_error() -> None:
    set_global_seed(42)


def test_set_global_seed_is_repeatable() -> None:
    set_global_seed(99)
    set_global_seed(99)


def test_set_global_seed_accepts_zero() -> None:
    set_global_seed(0)
