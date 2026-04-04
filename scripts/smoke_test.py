from pathlib import Path

from dotenv import load_dotenv

from rag_evo.generation.config import DecodingConfig
from rag_evo.generation.wrapper import GeneratorWrapper

# Explicitly point to the .env file in the project root
load_dotenv(Path(__file__).parent.parent / ".env")

wrapper = GeneratorWrapper()
output = wrapper.generate(
    system_prompt="You are a helpful assistant. Answer based on the context provided.",
    context_block="[DOC_1]: The Eiffel Tower is located in Paris, France.",
    query="Where is the Eiffel Tower?",
    decoding=DecodingConfig(temperature=0.0, max_tokens=100, seed=42),
)
print(f"Answer: {output.answer}")
print(f"Input tokens: {output.input_tokens}")
print(f"Output tokens: {output.output_tokens}")
print(f"Latency: {output.latency_ms:.1f}ms")
print(f"Provider: {output.provider}")
