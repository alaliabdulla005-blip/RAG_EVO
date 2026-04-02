"""Tests for src/rag_evo/utils/paths.py"""

import pytest

from rag_evo.utils.paths import ProjectPaths, get_project_paths


def test_get_project_paths_returns_dataclass() -> None:
    paths = get_project_paths()
    assert isinstance(paths, ProjectPaths)


def test_all_directories_exist_after_init() -> None:
    paths = get_project_paths()
    for field_name, value in paths.__dict__.items():
        assert value.exists(), f"{field_name} does not exist: {value}"


def test_custom_root_works(tmp_path) -> None:
    paths = get_project_paths(root=tmp_path)
    assert isinstance(paths, ProjectPaths)
    assert paths.root == tmp_path.resolve()


def test_missing_root_raises() -> None:
    from pathlib import Path

    with pytest.raises(FileNotFoundError):
        get_project_paths(root=Path("/nonexistent/path/that/does/not/exist"))
