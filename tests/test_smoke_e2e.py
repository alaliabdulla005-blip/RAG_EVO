"""End-to-end smoke tests for Phase 2.5 — no real API calls."""

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rag_evo.generation.wrapper import WrapperOutput


def _fixed_output() -> WrapperOutput:
    return WrapperOutput(
        answer="The capital of France is Paris. [DOC_2]",
        input_tokens=60,
        output_tokens=12,
        latency_ms=200.0,
        raw_prompt="Context:\n[DOC_2]: Paris is capital.\n\nQuestion: Capital?",
        raw_answer="The capital of France is Paris. [DOC_2]",
        model_id="test-model",
        provider="groq",
    )


@pytest.fixture()
def smoke_result(tmp_path: Path) -> dict:
    from scripts.run_smoke_test import run

    mock_wrapper = MagicMock()
    mock_wrapper.generate.return_value = _fixed_output()

    result = run(mock_wrapper)
    out_path = tmp_path / "smoke_test_output.json"
    out_path.write_text(json.dumps(result, indent=2))
    return {"result": result, "path": out_path}


def test_smoke_output_file_created(smoke_result: dict) -> None:
    assert smoke_result["path"].exists()


def test_smoke_output_has_required_fields(smoke_result: dict) -> None:
    data = json.loads(smoke_result["path"].read_text())
    required = {
        "phase",
        "question",
        "manually_selected_chunk",
        "answer",
        "cited_chunk_ids",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "model_id",
        "provider",
        "raw_prompt",
        "raw_answer",
    }
    assert required <= data.keys()


def test_smoke_cited_chunk_parsing() -> None:
    matches = re.findall(r"\[DOC_\d+\]", "The answer is in [DOC_2].")
    assert matches == ["[DOC_2]"]


def test_smoke_cited_chunk_parsing_empty() -> None:
    matches = re.findall(r"\[DOC_\d+\]", "No citations here.")
    assert matches == []
