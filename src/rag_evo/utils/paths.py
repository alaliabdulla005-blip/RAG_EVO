"""Centralised path management for the RAG-Evo project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectPaths:
    """Centralised path management for the RAG-Evo project.

    All paths are resolved as absolute paths from the project root.
    Subdirectories are created on first access if they do not exist.
    """

    root: Path
    data_raw: Path
    data_processed: Path
    data_indexes: Path
    results: Path
    logs: Path
    configs: Path
    docs: Path


def _find_project_root(start: Path) -> Path:
    """Walk up from *start* until pyproject.toml is found.

    Args:
        start: Directory to begin search from.

    Returns:
        Directory containing pyproject.toml.

    Raises:
        FileNotFoundError: If pyproject.toml is not found up to the filesystem root.
    """
    current = start.resolve()
    while True:
        if (current / "pyproject.toml").exists():
            return current
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(
                "Could not locate pyproject.toml while walking up from "
                f"{start.resolve()}"
            )
        current = parent


def get_project_paths(root: Path | None = None) -> ProjectPaths:
    """Build ProjectPaths from the given root or auto-detect from this file's location.

    Args:
        root: Explicit project root. If None, detected by walking up from this
            file until pyproject.toml is found.

    Returns:
        ProjectPaths with all subdirectories created.

    Raises:
        FileNotFoundError: If the resolved root does not exist.
    """
    if root is None:
        resolved_root = _find_project_root(Path(__file__).parent)
    else:
        resolved_root = root.resolve()
        if not resolved_root.exists():
            raise FileNotFoundError(
                f"Project root does not exist: {resolved_root}"
            )

    paths = ProjectPaths(
        root=resolved_root,
        data_raw=resolved_root / "data" / "raw",
        data_processed=resolved_root / "data" / "processed",
        data_indexes=resolved_root / "data" / "indexes",
        results=resolved_root / "results",
        logs=resolved_root / "logs",
        configs=resolved_root / "configs",
        docs=resolved_root / "docs",
    )

    for field_name, value in paths.__dict__.items():
        if field_name != "root":
            value.mkdir(parents=True, exist_ok=True)

    return paths
