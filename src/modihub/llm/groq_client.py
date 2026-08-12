import os
import logging
from collections.abc import Awaitable, Callable, Sequence
from functools import wraps
from typing import Any, TypeVar

import groq
import tenacity
from groq import AsyncGroq

from modihub.llm.base import LLMClient, LLMSchema

logging.basicConfig(level=logging.INFO)

T = TypeVar("T")


def api_exception_handler(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Normalize Groq SDK exceptions raised by asynchronous calls.

    Args:
        func: Asynchronous SDK operation wrapper.

    Returns:
        Wrapped coroutine function with normalized exception handling.
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        """Run the wrapped SDK call and normalize provider exceptions.

        Args:
            args: Positional arguments passed to the wrapped function.
            kwargs: Keyword arguments passed to the wrapped function.

        Returns:
            The wrapped function result.

        Raises:
            RuntimeError: If Groq returns an SDK error.
        """
        try:
            return await func(*args, **kwargs)
        except groq.APIConnectionError as e:
            logging.error(f"The server could not be reached: {e.__cause__}")
            raise RuntimeError(f"The server could not be reached: {e.__cause__}") from e
        except groq.RateLimitError as e:
            logging.error(f"The rate limit has been exceeded: {e.__cause__}")
            raise RuntimeError(f"The rate limit has been exceeded: {e.__cause__}") from e
        except groq.APIStatusError as e:
            logging.error(f"The server returned an error: {e.status_code}@{e.response}")
            raise RuntimeError(f"The server returned an error: {e.status_code}@{e.response}") from e
    return wrapper


class GroqClient(LLMClient):
    """Asynchronous client wrapper for Groq models.

    Args:
        model_name: Groq model identifier to use for chat completions.
        args: Positional arguments forwarded to `AsyncGroq`.
        kwargs: Keyword arguments forwarded to `AsyncGroq`.

    Attributes:
        system_instruction: Optional system prompt prepended to each request.
        api_client: Authenticated asynchronous Groq SDK client.
    """

    def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the Groq client.

        Args:
            model_name: Groq model identifier to use for chat completions.
            args: Positional arguments forwarded to `AsyncGroq`.
            kwargs: Keyword arguments forwarded to `AsyncGroq`.
        """
        super().__init__(model_name)
        self.system_instruction = kwargs.pop("system_instruction", "")
        self.api_client = self._get_api_client(*args, **kwargs)

    @staticmethod
    def _get_api_client(*args: Any, **kwargs: Any) -> AsyncGroq:
        """Create an authenticated asynchronous Groq SDK client.

        Args:
            args: Positional arguments forwarded to `AsyncGroq`.
            kwargs: Keyword arguments forwarded to `AsyncGroq`.

        Returns:
            Authenticated asynchronous Groq client.

        Raises:
            ValueError: If `GROQ_API_KEY` is not configured.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        return AsyncGroq(api_key=api_key, *args, **kwargs)

    @tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3), reraise=True)
    @api_exception_handler
    async def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        """Generate text from a Groq chat model.

        Args:
            prompt: User prompt to send to the model.
            args: Reserved positional generation arguments.
            kwargs: Groq chat completion parameters such as `temperature`.

        Returns:
            Generated text content, or an empty string if the response has no content.
        """
        messages_queue = (
            [{"role": "system", "content": self.system_instruction}]
            if self.system_instruction
            else []
        )
        messages_queue.append({"role": "user", "content": prompt})
        try:
            completion = await self.api_client.chat.completions.create(
                model=self.model_name, messages=messages_queue, **kwargs
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            logging.error(f"Error generating content: {e}")
            raise

    @classmethod
    async def list_models(cls) -> Sequence[LLMSchema]:
        """List Groq models available to the configured API key.

        Returns:
            Sequence of Groq model metadata.
        """
        api_client = cls._get_api_client()
        try:
            available_models = (await api_client.models.list()).data
            return [
                LLMSchema(
                    client="groq",
                    name=model.id,
                    display_name=model.id
                )
                for model in available_models
            ]
        except Exception as e:
            logging.error(f"Failed to list models: {e}")
            return []

if __name__ == "__main__":
    import asyncio

    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())

    async def main() -> None:
        """Run a small manual Groq smoke test."""
        models = await GroqClient.list_models()
        for model in models:
            print(model.name)

        try:
            client = GroqClient.get_model("llama3-8b-8192", system_instruction="Generate the output in markdown format")
            response = await client.generate("Who are you?")
            print(response)
        except Exception as e:
            logging.error(f"Error during execution: {e}")

    asyncio.run(main())
