# Provider Clients

MODIHUB includes provider wrappers behind the shared `LLM` factory. Each client exposes the same asynchronous call pattern, while still accepting provider-specific keyword arguments for generation calls.

## Supported Providers

| Provider | Client class | Credential |
| --- | --- | --- |
| OpenAI | `OpenAIClient` | `OPENAI_API_KEY` |
| Google Gemini | `GeminiClient` | `GEMINI_API_KEY` |
| Anthropic | `AnthropicClient` | `ANTHROPIC_API_KEY` |
| Groq | `GroqClient` | `GROQ_API_KEY` |
| Ollama | `OllamaClient` | Local Ollama server |

## Text Prompts

```python
llm = await LLM.create("gpt-4o-mini")
response = await llm("Write a two-line project summary.")
```

Provider-specific generation options can be passed to the call:

```python
response = await llm(
    "Write a short abstract.",
    temperature=0.2,
)
```

## System Instructions

All bundled clients accept `system_instruction` when the model client is created:

```python
llm = await LLM.create(
    "gpt-4o-mini",
    system_instruction="Answer for a scientific computing audience.",
)
```

## Multimodal Prompts

OpenAI and Ollama normalize PIL images into provider message formats. Gemini also accepts multimodal content through its SDK.

```python
import asyncio

from PIL import Image
from dotenv import find_dotenv, load_dotenv
from modihub.llm import LLM


async def main() -> None:
    load_dotenv(find_dotenv())

    image = Image.open("image.png")
    llm = await LLM.create("models/gemini-1.5-flash-8b")
    response = await llm(["Describe this image.", image])
    print(response)


asyncio.run(main())
```

## Direct Client Use

The provider classes can be imported directly when a caller already knows which provider is needed:

```python
from modihub.llm import OpenAIClient

client = OpenAIClient.get_model("gpt-4o-mini")
response = await client.generate("Generate a short title.")
```
