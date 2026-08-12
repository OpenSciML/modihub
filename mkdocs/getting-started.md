# Getting Started

## Installation

Install the package from PyPI:

```bash
pip install -U modihub
```

For local development from this repository:

```bash
uv sync
```

## Configure Provider Credentials

MODIHUB reads provider credentials from environment variables. A local `.env` file is supported through `python-dotenv` in examples.

```bash
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_API_KEY
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

Ollama uses a locally running Ollama server and does not require a hosted-provider API key.

## List Available Models

Model discovery is asynchronous because most providers require an API call.

```python
import asyncio

from dotenv import find_dotenv, load_dotenv
from modihub.llm import LLM


async def main() -> None:
    load_dotenv(find_dotenv())

    available_models = await LLM.available_models()
    for provider, models in available_models.group_by("client"):
        print(provider)
        for model in models:
            print(f"  {model.name}")


asyncio.run(main())
```

## Generate Text

Create a model client with `await LLM.create(...)`, then call it with `await llm(...)`.

```python
import asyncio

from dotenv import find_dotenv, load_dotenv
from modihub.llm import LLM


async def main() -> None:
    load_dotenv(find_dotenv())

    llm = await LLM.create(
        "gpt-4o-mini",
        system_instruction="Respond in concise technical prose.",
    )
    response = await llm("Explain what an async LLM client is.")
    print(response)


asyncio.run(main())
```

## Error Handling

`LLM.create(...)` checks registered providers and raises `ValueError` when the requested model is not available from any configured client. Provider SDK failures are raised by the underlying client or translated to `RuntimeError` where the provider wrapper does explicit normalization.
