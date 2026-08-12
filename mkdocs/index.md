# MODIHUB

MODIHUB provides one asynchronous interface for working with LLM providers such as OpenAI, Gemini, Anthropic, Ollama, and Groq. It is designed for code that needs to switch between providers, inspect available models, pass text or multimodal prompts, and evaluate outputs with simple text metrics.

## What It Provides

- Async model discovery through `LLM.available_models()`.
- Async model creation through `LLM.create(...)`.
- A shared callable client interface: `await llm(prompt)`.
- Provider clients for OpenAI, Google Gemini, Anthropic, Ollama, and Groq.
- Basic evaluation utilities for scoring generated text.

## Minimal Example

```python
import asyncio

from dotenv import find_dotenv, load_dotenv
from modihub.llm import LLM


async def main() -> None:
    load_dotenv(find_dotenv())

    llm = await LLM.create("gpt-4o-mini")
    response = await llm("Summarize the purpose of MODIHUB in one sentence.")
    print(response)


asyncio.run(main())
```

## Documentation

- [Getting Started](getting-started.md) covers installation, API keys, and the first request.
- [Provider Clients](providers.md) explains provider-specific behavior and multimodal prompts.
- [Evaluation](evaluation.md) shows how to score model outputs.
- [API Reference](api-reference.md) summarizes the public classes and functions.
- [Development](development.md) documents local checks and docs builds.
