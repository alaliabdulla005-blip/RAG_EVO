"""Tests for src/rag_evo/utils/logger.py"""

import logging

from rag_evo.utils.logger import get_logger


def test_get_logger_returns_logger(tmp_log_dir) -> None:
    log = get_logger("test_module", tmp_log_dir, "run_001")
    assert isinstance(log, logging.Logger)


def test_log_file_created(tmp_log_dir) -> None:
    get_logger("test_file", tmp_log_dir, "run_002")
    log_file = tmp_log_dir / "run_002_test_file.log"
    assert log_file.exists()


def test_no_duplicate_handlers(tmp_log_dir) -> None:
    log1 = get_logger("test_dedup", tmp_log_dir, "run_003")
    initial_count = len(log1.handlers)
    log2 = get_logger("test_dedup", tmp_log_dir, "run_003")
    assert len(log2.handlers) == initial_count
