"""Logging utility for RAG-Evo: dual stdout/file logger with timestamped files."""

import logging
from pathlib import Path

_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, log_dir: Path, run_id: str) -> logging.Logger:
    """Create a logger writing to both stdout and a timestamped file.

    Args:
        name: Logger name, typically the calling module's __name__.
        log_dir: Directory where the log file will be written.
        run_id: Unique identifier for this run, used in the log filename.

    Returns:
        Configured Logger instance.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_id}_{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
