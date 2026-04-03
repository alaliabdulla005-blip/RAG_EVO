"""Tests for src/rag_evo/generation/wrapper.py and config.py"""

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import groq
import pytest

from rag_evo.generation.config import DecodingConfig
from rag_evo.generation.wrapper import GeneratorWrapper, WrapperOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(
    content: str = "Paris, France.",
    input_tokens: int = 50,
    output_tokens: int = 10,
) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = input_tokens
    resp.usage.completion_tokens = output_tokens
    return resp


def _rate_limit_error() -> groq.RateLimitError:
    mock_http = MagicMock()
    mock_http.status_code = 429
    mock_http.request = MagicMock()
    return groq.RateLimitError("rate limited", response=mock_http, body=None)


@pytest.fixture()
def wrapper() -> GeneratorWrapper:
    return GeneratorWrapper(api_key="test-key", model_id="test-model")


# ---------------------------------------------------------------------------
# DecodingConfig tests
# ---------------------------------------------------------------------------


def test_decoding_config_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError):
        DecodingConfig(temperature=2.0)


def test_decoding_config_rejects_nondefault_top_p() -> None:
    with pytest.raises(ValueError):
        DecodingConfig(top_p=0.9)


# ---------------------------------------------------------------------------
# WrapperOutput tests
# ---------------------------------------------------------------------------


def test_wrapper_output_is_frozen() -> None:
    out = WrapperOutput(
        answer="Paris",
        input_tokens=10,
        output_tokens=5,
        latency_ms=100.0,
        raw_prompt="Context:\nfoo\n\nQuestion: bar",
        raw_answer="Paris",
        model_id="test-model",
        provider="groq",
    )
    with pytest.raises(FrozenInstanceError):
        out.answer = "London"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GeneratorWrapper.generate tests
# ---------------------------------------------------------------------------


def test_generate_returns_wrapper_output(wrapper: GeneratorWrapper) -> None:
    mock_resp = _mock_response("Paris, France.")
    with patch.object(
        wrapper._client.chat.completions, "create", return_value=mock_resp
    ):
        output = wrapper.generate(
            system_prompt="Answer based on context.",
            context_block="[DOC_1]: The Eiffel Tower is in Paris.",
            query="Where is the Eiffel Tower?",
        )

    assert isinstance(output, WrapperOutput)
    assert output.answer == "Paris, France."
    assert output.input_tokens == 50
    assert output.output_tokens == 10
    assert output.latency_ms >= 0.0
    assert output.provider == "groq"


def test_generate_constructs_prompt_correctly(wrapper: GeneratorWrapper) -> None:
    system_prompt = "You are a helpful assistant."
    context_block = "[DOC_1]: Eiffel Tower is in Paris."
    query = "Where is it?"

    with patch.object(
        wrapper._client.chat.completions,
        "create",
        return_value=_mock_response("Paris."),
    ) as mock_create:
        wrapper.generate(
            system_prompt=system_prompt,
            context_block=context_block,
            query=query,
        )

    messages = mock_create.call_args[1]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == system_prompt
    assert messages[1]["role"] == "user"
    user_content: str = messages[1]["content"]
    assert f"Context:\n{context_block}" in user_content
    assert f"Question: {query}" in user_content


def test_generate_retries_on_rate_limit() -> None:
    wrapper = GeneratorWrapper(
        api_key="test-key", model_id="test-model", max_retries=1
    )
    err = _rate_limit_error()
    call_count = 0

    def _side_effect(*args: object, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise err
        return _mock_response("Paris.")

    with patch.object(
        wrapper._client.chat.completions, "create", side_effect=_side_effect
    ):
        with patch("rag_evo.generation.wrapper.time.sleep"):
            output = wrapper.generate(
                system_prompt="sys",
                context_block="ctx",
                query="q",
            )

    assert isinstance(output, WrapperOutput)
    assert call_count == 2


def test_generate_raises_after_max_retries() -> None:
    wrapper = GeneratorWrapper(
        api_key="test-key", model_id="test-model", max_retries=2
    )
    err = _rate_limit_error()
    call_count = 0

    def _always_fail(*args: object, **kwargs: object) -> None:
        nonlocal call_count
        call_count += 1
        raise err

    with patch.object(
        wrapper._client.chat.completions, "create", side_effect=_always_fail
    ):
        with patch("rag_evo.generation.wrapper.time.sleep"):
            with pytest.raises(RuntimeError):
                wrapper.generate(
                    system_prompt="sys", context_block="ctx", query="q"
                )

    assert call_count == 3  # initial + 2 retries


def test_generate_raises_on_empty_answer(wrapper: GeneratorWrapper) -> None:
    with patch.object(
        wrapper._client.chat.completions, "create", return_value=_mock_response("")
    ):
        with pytest.raises(ValueError, match="empty answer"):
            wrapper.generate(system_prompt="sys", context_block="ctx", query="q")
