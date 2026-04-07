"""GeneratorWrapper: sole interface between the RAG pipeline and the LLM."""

import os
import time
from dataclasses import dataclass
from datetime import datetime

import groq
import openai
from dotenv import load_dotenv

from rag_evo.generation.config import DecodingConfig
from rag_evo.utils.logger import get_logger
from rag_evo.utils.paths import get_project_paths

load_dotenv()

_DEFAULT_GROQ_MODEL_ID = "llama-3.1-8b-instant"
_DEFAULT_SAMBANOVA_MODEL_ID = "Meta-Llama-3.1-8B-Instruct"
_DEFAULT_SAMBANOVA_BASE_URL = "https://api.sambanova.ai/v1"


@dataclass(frozen=True)
class WrapperOutput:
    """Structured output from a single generation call.

    All fields are mandatory. No field may be None.

    Args:
        answer: The model's generated answer text, stripped of leading/trailing
            whitespace.
        input_tokens: Number of tokens in the prompt as reported by the API.
        output_tokens: Number of tokens in the generated answer as reported by the API.
        latency_ms: Wall-clock time for the API call in milliseconds.
        raw_prompt: The exact prompt string sent to the API.
        raw_answer: The exact answer string returned by the API before any processing.
        model_id: The model identifier used for this call.
        provider: The inference provider used (e.g. "groq", "sambanova").
    """

    answer: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw_prompt: str
    raw_answer: str
    model_id: str
    provider: str


class GeneratorWrapper:
    """Wrapper around Llama-3.1-8B-Instruct via Groq or SambaNova inference API.

    This is the sole interface between the RAG pipeline and the LLM.
    No other module should import or call the inference client directly.

    The provider is selected via the INFERENCE_PROVIDER environment variable.
    Supported values: "groq" (default), "sambanova".

    Args:
        api_key: API key override. If None, read from provider-specific env var.
        model_id: Model identifier override. If None, read from
            provider-specific env var.
        default_decoding: Default DecodingConfig to use when none is provided per call.
        max_retries: Number of times to retry on rate-limit errors (429) before raising.
        retry_base_delay: Base delay in seconds for exponential backoff on retries.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str | None = None,
        default_decoding: DecodingConfig | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        self._default_decoding = default_decoding or DecodingConfig()
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay

        self._provider = os.getenv("INFERENCE_PROVIDER", "groq").lower()

        if self._provider == "sambanova":
            resolved_key = api_key or os.environ.get(
                "SAMBANOVA_API_KEY", ""
            )
            self._model_id = model_id or os.environ.get(
                "SAMBANOVA_MODEL_ID", _DEFAULT_SAMBANOVA_MODEL_ID
            )
            base_url = os.environ.get(
                "SAMBANOVA_BASE_URL", _DEFAULT_SAMBANOVA_BASE_URL
            )
            self._client = openai.OpenAI(
                api_key=resolved_key, base_url=base_url
            )
        else:
            # Default: groq
            self._provider = "groq"
            resolved_key = api_key or os.environ.get("GROQ_API_KEY", "")
            self._model_id = model_id or os.environ.get(
                "GROQ_MODEL_ID", _DEFAULT_GROQ_MODEL_ID
            )
            self._client = groq.Groq(api_key=resolved_key)

        run_id = "wrapper_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = get_project_paths()
        self._logger = get_logger(__name__, paths.logs, run_id)

    @property
    def provider(self) -> str:
        """Return the active provider string."""
        return self._provider

    def generate(
        self,
        system_prompt: str,
        context_block: str,
        query: str,
        decoding: DecodingConfig | None = None,
    ) -> WrapperOutput:
        """Generate an answer from the model given a system prompt, context, and query.

        The final prompt sent to the API is constructed as:
            system: {system_prompt}
            user: Context:\\n{context_block}\\n\\nQuestion: {query}

        Args:
            system_prompt: Instructions for the model's behaviour and citation format.
            context_block: Retrieved passages, pre-formatted with chunk labels.
            query: The user's question.
            decoding: Decoding configuration. Uses default_decoding if None.

        Returns:
            WrapperOutput with all fields populated.

        Raises:
            RuntimeError: If all retries are exhausted.
            ValueError: If the API returns an empty answer.
        """
        cfg = decoding if decoding is not None else self._default_decoding
        user_message = f"Context:\n{context_block}\n\nQuestion: {query}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        self._logger.debug("Calling model=%s", self._model_id)

        response = None
        latency_ms = 0.0

        # Build kwargs — SambaNova does not support seed
        call_kwargs: dict = {
            "model": self._model_id,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "top_p": cfg.top_p,
        }
        if self._provider == "groq":
            call_kwargs["seed"] = cfg.seed

        rate_limit_exc: type = (
            openai.RateLimitError
            if self._provider == "sambanova"
            else groq.RateLimitError
        )

        for attempt in range(self._max_retries + 1):
            try:
                t0 = time.perf_counter()
                response = self._client.chat.completions.create(
                    **call_kwargs
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                break
            except rate_limit_exc as exc:
                if attempt < self._max_retries:
                    delay = self._retry_base_delay * (2**attempt)
                    self._logger.warning(
                        "Rate limit hit, attempt %d/%d. "
                        "Retrying in %.1fs.",
                        attempt + 1,
                        self._max_retries + 1,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    raise RuntimeError(
                        f"{self._provider} rate limit: "
                        f"all {self._max_retries + 1} "
                        "attempts exhausted."
                    ) from exc

        raw_answer = response.choices[0].message.content or ""  # type: ignore[union-attr]
        answer = raw_answer.strip()

        if not answer:
            raise ValueError(
                "Model returned empty answer. "
                "Check prompt construction and API connectivity."
            )

        self._logger.info(
            "model=%s input_tokens=%d output_tokens=%d latency_ms=%.1f",
            self._model_id,
            response.usage.prompt_tokens,  # type: ignore[union-attr]
            response.usage.completion_tokens,  # type: ignore[union-attr]
            latency_ms,
        )
        self._logger.debug(
            "raw_prompt=%r raw_answer=%r", user_message, raw_answer
        )

        return WrapperOutput(
            answer=answer,
            input_tokens=response.usage.prompt_tokens,  # type: ignore[union-attr]
            output_tokens=response.usage.completion_tokens,  # type: ignore[union-attr]
            latency_ms=latency_ms,
            raw_prompt=user_message,
            raw_answer=raw_answer,
            model_id=self._model_id,
            provider=self._provider,
        )
