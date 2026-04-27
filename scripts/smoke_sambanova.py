""""Smoke test for GeneratorWrapper — originally used with
SambaNova (deprecated April 2026); now runs against
INFERENCE_PROVIDER env var."
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

from rag_evo.generation.config import DecodingConfig
from rag_evo.generation.wrapper import GeneratorWrapper

wrapper = GeneratorWrapper()
output = wrapper.generate(
    system_prompt=(
        "You are a helpful assistant. Answer based on context."
    ),
    context_block=(
        "[DOC_1]: The Eiffel Tower is located in Paris, France."
    ),
    query="Where is the Eiffel Tower?",
    decoding=DecodingConfig(temperature=0.0, max_tokens=50, seed=42),
)
print(f"Provider: {output.provider}")
print(f"Answer: {output.answer}")
print(f"Input tokens: {output.input_tokens}")
print(f"Output tokens: {output.output_tokens}")
print(f"Latency: {output.latency_ms:.1f}ms")
