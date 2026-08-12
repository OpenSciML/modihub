from collections.abc import Sequence
from typing import Any

import ollama
import tenacity
from PIL.Image import Image as PILImage

from modihub.llm.base import LLMClient, LLMSchema
from modihub.utils import ImageUtils


class OllamaClient(LLMClient):
    """Asynchronous client wrapper for locally hosted Ollama models.

    Args:
        model_name: Ollama model identifier to use for chat requests.
        args: Positional arguments forwarded to `ollama.AsyncClient`.
        kwargs: Keyword arguments forwarded to `ollama.AsyncClient`.

    Attributes:
        system_instruction: Optional system prompt prepended to each request.
        api_client: Asynchronous Ollama client.
    """

    def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the Ollama client.

        Args:
            model_name: Ollama model identifier to use for chat requests.
            args: Positional arguments forwarded to `ollama.AsyncClient`.
            kwargs: Keyword arguments forwarded to `ollama.AsyncClient`.
        """
        super().__init__(model_name)
        self.system_instruction = kwargs.pop("system_instruction", "")
        self.api_client = OllamaClient.get_api_client(*args, **kwargs)

    @staticmethod
    def get_api_client(*args: Any, **kwargs: Any) -> ollama.AsyncClient:
        """Create an asynchronous Ollama API client.

        Args:
            args: Positional arguments forwarded to `ollama.AsyncClient`.
            kwargs: Keyword arguments forwarded to `ollama.AsyncClient`.

        Returns:
            Asynchronous Ollama API client.
        """
        return ollama.AsyncClient(*args, **kwargs)

    @staticmethod
    def _normalized_prompt_content(prompt: Any) -> dict[str, Any]:
        """Normalize the prompt to an Ollama message dictionary.

        Supports:
            - Single string prompt
            - Single PIL Image
            - List of mixed strings and PIL Images

        Args:
            prompt: String, PIL image, or mixed list prompt.

        Returns:
            Message dictionary with role, content, and optionally images.

        Raises:
            ValueError: If the input type is not supported.
        """
        if isinstance(prompt, str):
            return {"role": "user", "content": prompt}

        elif isinstance(prompt, PILImage):
            return {
                "role": "user",
                "content": "",
                "images": [ImageUtils.image_to_base64_url(prompt)],
            }

        elif isinstance(prompt, list):
            images = [img for img in prompt if isinstance(img, PILImage)]
            texts = [txt for txt in prompt if isinstance(txt, str)]
            return {
                "role": "user",
                "content": "".join(texts),
                "images": [ImageUtils.image_to_base64(img) for img in images],
            }

        raise ValueError("Unsupported prompt type")

    @tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3), reraise=True)
    async def generate(self, prompt: Any, *args: Any, **kwargs: Any) -> str:
        """Generate a response from the Ollama model.

        Args:
            prompt: Prompt content; can be a string, PIL image, or list of both.
            args: Reserved positional generation arguments.
            kwargs: Ollama chat request options.

        Returns:
            Response content from the model.
        """
        messages_queue = (
            [{"role": "system", "content": self.system_instruction}]
            if self.system_instruction
            else []
        )
        normalized_prompt = self._normalized_prompt_content(prompt)
        messages_queue.append(normalized_prompt)

        response = await self.api_client.chat(
            model=self.model_name,
            messages=messages_queue,
            **kwargs
        )

        ai_message = response.get("message", {})
        return ai_message.get("content", "")

    @staticmethod
    async def list_models(*args: Any, **kwargs: Any) -> Sequence[LLMSchema]:
        """List all available models from the local Ollama server.

        Args:
            args: Positional arguments forwarded to the client.
            kwargs: Keyword arguments forwarded to the client.

        Returns:
            Sequence of Ollama model metadata.
        """
        api_client = OllamaClient.get_api_client(*args, **kwargs)
        ollama_models = await api_client.list()
        return [
            LLMSchema(
                name=model_info["model"],
                display_name=model_info["model"],
                client="ollama"
            )
            for model_info in ollama_models["models"]
        ]


if __name__ == '__main__':
    import asyncio

    async def main() -> None:
        """Run a small manual Ollama smoke test."""
        print("Available Ollama Models:")
        for model in await OllamaClient.list_models():
            print(f"- {model.name}")

        client = OllamaClient.get_model("llama3.1:latest", system_instruction="Generate the output in markdown format")

        response = await client.generate("Who are you?")
        print("\nModel Response:\n")
        print(response)

    asyncio.run(main())
