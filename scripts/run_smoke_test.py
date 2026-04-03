"""Feasibility smoke test — Phase 2.5.

Wires the GeneratorWrapper to manually prepared dummy inputs.
Saves output to results/smoke/smoke_test_output.json.
Run from project root: python scripts/run_smoke_test.py
"""

import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from rag_evo.generation.config import DecodingConfig  # noqa: E402
from rag_evo.generation.wrapper import GeneratorWrapper  # noqa: E402

QUESTION = "What is the capital of France?"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based only on the "
    "provided context. Cite the document you used as [DOC_N]."
)
CONTEXT_BLOCK = "[DOC_2]: Paris is the capital and largest city of France."
MANUALLY_SELECTED_CHUNK = "[DOC_2]: Paris is the capital and largest city of France."


def run(wrapper: GeneratorWrapper) -> dict:
    output = wrapper.generate(
        system_prompt=SYSTEM_PROMPT,
        context_block=CONTEXT_BLOCK,
        query=QUESTION,
        decoding=DecodingConfig(temperature=0.0, max_tokens=150, seed=42),
    )
    cited_chunk_ids = re.findall(r"\[DOC_\d+\]", output.answer)
    return {
        "phase": "2.5",
        "question": QUESTION,
        "manually_selected_chunk": MANUALLY_SELECTED_CHUNK,
        "answer": output.answer,
        "cited_chunk_ids": cited_chunk_ids,
        "input_tokens": output.input_tokens,
        "output_tokens": output.output_tokens,
        "latency_ms": output.latency_ms,
        "model_id": output.model_id,
        "provider": output.provider,
        "raw_prompt": output.raw_prompt,
        "raw_answer": output.raw_answer,
    }


def main() -> int:
    try:
        wrapper = GeneratorWrapper()
        result = run(wrapper)

        out_path = Path("results/smoke/smoke_test_output.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))

        print("Phase 2.5 smoke test passed.")
        print(f"Answer: {result['answer']}")
        print(f"Cited chunks: {result['cited_chunk_ids']}")
        print("Artifact saved to: results/smoke/smoke_test_output.json")
        return 0
    except Exception as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
