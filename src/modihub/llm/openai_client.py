import itertools
import os
from collections.abc import Sequence
from typing import Any

import openai
import tenacity
from PIL.Image import Image as PILImage

from modihub.llm.base import LLMClient, LLMSchema
from modihub.utils import ImageUtils


class OpenAIClient(LLMClient):
    """Asynchronous client wrapper for OpenAI chat models.

    Args:
        model_name: OpenAI model identifier to use for chat completions.
        args: Positional arguments forwarded to `openai.AsyncOpenAI`.
        kwargs: Keyword arguments forwarded to `openai.AsyncOpenAI`.

    Attributes:
        system_instruction: Optional system prompt prepended to each request.
        api_client: Authenticated asynchronous OpenAI SDK client.
    """

    def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the OpenAI client.

        Args:
            model_name: OpenAI model identifier to use for chat completions.
            args: Positional arguments forwarded to `openai.AsyncOpenAI`.
            kwargs: Keyword arguments forwarded to `openai.AsyncOpenAI`.
        """
        super().__init__(model_name)
        self.system_instruction = kwargs.pop("system_instruction", "")
        self.api_client = OpenAIClient.get_api_client(*args, **kwargs)

    @staticmethod
    def get_api_client(*args: Any, **kwargs: Any) -> openai.AsyncOpenAI:
        """Create an authenticated asynchronous OpenAI SDK client.

        Args:
            args: Positional arguments forwarded to `openai.AsyncOpenAI`.
            kwargs: Keyword arguments forwarded to `openai.AsyncOpenAI`.

        Returns:
            Authenticated asynchronous OpenAI client.

        Raises:
            ValueError: If `OPENAI_API_KEY` is not configured.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return openai.AsyncOpenAI(api_key=api_key, *args, **kwargs)

    def _normalized_prompt(self, prompt: Any) -> list[dict[str, Any]]:
        """Normalize text and image prompts for OpenAI chat messages.

        Args:
            prompt: String, PIL image, or list containing either type.

        Returns:
            OpenAI-compatible content parts.

        Raises:
            ValueError: If the prompt type is unsupported.
        """
        if isinstance(prompt, str):
            return [{"type": "text", "text": prompt}]
        elif isinstance(prompt, PILImage):
            return [
                {
                    "type": "image_url",
                    "image_url": {"url": ImageUtils.image_to_base64_url(prompt)},
                }
            ]
        elif isinstance(prompt, list):
            return list(
                itertools.chain.from_iterable(
                    self._normalized_prompt(p) for p in prompt
                )
            )
        else:
            raise ValueError("Unsupported prompt type")

    @tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3), reraise=True)
    async def generate(self, prompt: Any, *args: Any, **kwargs: Any) -> str:
        """Generate text from an OpenAI chat model.

        Args:
            prompt: String, PIL image, or mixed list prompt to send to the model.
            args: Reserved positional generation arguments.
            kwargs: OpenAI chat completion parameters such as `temperature`.

        Returns:
            Generated text content, or an empty string if the response has no content.
        """
        messages_queue = (
            [{"role": "system", "content": self.system_instruction}]
            if self.system_instruction
            else []
        )
        normalized_prompt = self._normalized_prompt(prompt)
        messages_queue.append({"role": "user", "content": normalized_prompt})
        completion = await self.api_client.chat.completions.create(
            model=self.model_name, messages=messages_queue, **kwargs
        )
        return completion.choices[0].message.content or ""

    @classmethod
    async def list_models(cls, *args: Any, **kwargs: Any) -> Sequence[LLMSchema]:
        """List OpenAI models available to the configured API key.

        Args:
            args: Positional arguments forwarded to the SDK client.
            kwargs: Keyword arguments forwarded to the SDK client.

        Returns:
            Sequence of OpenAI model metadata.
        """
        api_client = OpenAIClient.get_api_client(*args, **kwargs)
        available_models = await api_client.models.list()
        return [
            LLMSchema(name=model_info.id, display_name=model_info.id, client="openai")
            for model_info in available_models
        ]


if __name__ == '__main__':
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    import asyncio

    async def main() -> None:
        """Run a small manual OpenAI smoke test."""
        for m in await OpenAIClient.list_models():
            print(m.name)
        response = OpenAIClient.get_model("gpt-4o-mini", system_instruction="generate the output in markdown format")
        print(await response("who are you?"))

    asyncio.run(main())
