"""Shared pytest fixtures for RAG-Evo tests."""

import pytest


@pytest.fixture()
def tmp_log_dir(tmp_path):
    """Return a temporary directory for log files."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir
