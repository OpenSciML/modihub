import os
from collections.abc import Awaitable, Callable, Sequence
from functools import wraps
from typing import Any, TypeVar

import anthropic
import tenacity
from anthropic import AsyncAnthropic
from modihub.llm.base import LLMClient, LLMSchema

T = TypeVar("T")


def api_exception_handler(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Translate Anthropic SDK errors into runtime exceptions.

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
            RuntimeError: If Anthropic returns an SDK error.
        """
        try:
            return await func(*args, **kwargs)
        except anthropic.AnthropicError as e:
            raise RuntimeError(f"Anthropic API error: {e}") from e
        except anthropic.RateLimitError as e:
            raise RuntimeError(f"Rate limit exceeded: {e.__cause__}") from e

    return wrapper


class AnthropicClient(LLMClient):
    """Asynchronous client wrapper for Anthropic Claude models.

    Args:
        model_name: Anthropic model identifier to use for generation.
        args: Positional arguments forwarded to `AsyncAnthropic`.
        kwargs: Keyword arguments forwarded to `AsyncAnthropic`.

    Attributes:
        system_instruction: Optional system prompt sent with each request.
        api_client: Authenticated asynchronous Anthropic SDK client.
    """

    def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the Anthropic client.

        Args:
            model_name: Anthropic model identifier to use for generation.
            args: Positional arguments forwarded to `AsyncAnthropic`.
            kwargs: Keyword arguments forwarded to `AsyncAnthropic`.
        """
        super().__init__(model_name)
        self.system_instruction = kwargs.pop("system_instruction", "")
        self.api_client = self.get_api_client(*args, **kwargs)

    @staticmethod
    def get_api_client(*args: Any, **kwargs: Any) -> AsyncAnthropic:
        """Create an authenticated asynchronous Anthropic SDK client.

        Args:
            args: Positional arguments forwarded to `AsyncAnthropic`.
            kwargs: Keyword arguments forwarded to `AsyncAnthropic`.

        Returns:
            Authenticated asynchronous Anthropic client.

        Raises:
            ValueError: If `ANTHROPIC_API_KEY` is not configured.
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        return AsyncAnthropic(api_key=api_key, *args, **kwargs)

    @tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3), reraise=True)
    @api_exception_handler
    async def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        """Generate a response from Claude for a text prompt.

        Args:
            prompt: User prompt to send to Claude.
            args: Reserved positional generation arguments.
            kwargs: Anthropic message parameters such as `temperature`.

        Returns:
            Generated response text.
        """
        response = await self.api_client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            system=self.system_instruction or anthropic.NOT_GIVEN,
            max_tokens=kwargs.pop("max_tokens", 1024),
            **kwargs
        )

        if not response.content:
            return ""
        first_block = response.content[0]
        return getattr(first_block, "text", str(first_block))

    @classmethod
    async def list_models(cls, *args: Any, **kwargs: Any) -> Sequence[LLMSchema]:
        """List Anthropic models available to the configured API key.

        Args:
            args: Positional arguments forwarded to the SDK client.
            kwargs: Keyword arguments forwarded to the SDK client.

        Returns:
            Sequence of Anthropic model metadata.
        """
        api_client = cls.get_api_client(*args, **kwargs)
        models = await api_client.models.list()
        return [
            LLMSchema(
                name=model.id,
                display_name=model.display_name,
                client="anthropic"
            )
            for model in models
        ]


# === Example usage ===
if __name__ == "__main__":
    import asyncio

    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    async def main() -> None:
        """Run a small manual Anthropic smoke test."""
        print("Available Claude Models:")
        for model in await AnthropicClient.list_models():
            print(f"- {model.name}")

        model_instance = AnthropicClient.get_model(
            "claude-3-7-sonnet-20250219",
            system_instruction="Generate the output in markdown format."
        )

        try:
            print("\nClaude's Response:\n")
            print(await model_instance("Who are you?"))
        except Exception as e:
            print(f"Error: {e}")

    asyncio.run(main())
