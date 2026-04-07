from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(".env"))

import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("SAMBANOVA_API_KEY"),
    base_url=os.getenv("SAMBANOVA_BASE_URL"),
)

response = client.chat.completions.create(
    model=os.getenv("SAMBANOVA_MODEL_ID"),
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Answer based on the context provided."},
        {"role": "user", "content": "Context:\n[DOC_1]: The Eiffel Tower is located in Paris, France.\n\nQuestion: Where is the Eiffel Tower?"},
    ],
    temperature=0.0,
    max_tokens=100,
    seed=42,
)

print(f"Answer: {response.choices[0].message.content}")
print(f"Input tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Model: {response.model}")