import asyncio
import os
import logging
from collections.abc import Sequence
from typing import Any

import tenacity
from google import genai
from modihub.llm.base import LLMClient, LLMSchema
from google.genai.types import (
    GenerateImagesResponse,
    GenerateContentResponse,
    GenerateVideosOperation,
    EmbedContentResponse,
    GenerateContentConfigOrDict,
    GenerateContentConfig,
)

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    """Asynchronous client wrapper for Google Gemini APIs.

    Args:
        model_name: Gemini model identifier to use for generation.
        args: Positional arguments forwarded to `genai.Client`.
        kwargs: Keyword arguments forwarded to `genai.Client`.

    Attributes:
        system_instruction: Optional instruction attached to text generation config.
        api_client: Authenticated Google GenAI client with async resources under `.aio`.
    """

    def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the Gemini client with model name and optional instruction.

        Args:
            model_name: Gemini model identifier to use for generation.
            args: Positional arguments forwarded to `genai.Client`.
            kwargs: Keyword arguments forwarded to `genai.Client`.
        """
        super().__init__(model_name)
        self.model_name = model_name
        self.system_instruction = kwargs.pop("system_instruction", None)
        self.api_client = GeminiClient.get_api_client(*args, **kwargs)

    @staticmethod
    def get_api_client(*args: Any, **kwargs: Any) -> genai.Client:
        """Create an authenticated Google GenAI SDK client.

        Args:
            args: Positional arguments forwarded to `genai.Client`.
            kwargs: Keyword arguments forwarded to `genai.Client`.

        Returns:
            Configured Google GenAI client.

        Raises:
            ValueError: If `GEMINI_API_KEY` is not configured.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        return genai.Client(api_key=api_key, *args, **kwargs)

    @tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3), reraise=True)
    async def generate(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
        """Generate content from Gemini based on the requested modality.

        Supported modalities:
            - TEXT (default)
            - IMAGE
            - VIDEO
            - EMBEDDINGS

        Args:
            prompt: Prompt input to the model.
            args: Reserved positional generation arguments.
            kwargs: Gemini generation options, including optional `modality`.

        Returns:
            Generated content matching the requested modality.

        Raises:
            RuntimeError: If video generation fails.
            AssertionError: If prompt format is incorrect for given modality.
        """
        try:
            modality = kwargs.pop("modality", "TEXT").upper()

            if modality == "IMAGE":
                assert isinstance(prompt, str), "Prompt must be a string for image generation"
                response: GenerateImagesResponse = await self.api_client.aio.models.generate_images(
                    model=self.model_name,
                    prompt=prompt,
                    **kwargs
                )
                return response.generated_images

            elif modality == "VIDEO":
                assert isinstance(prompt, str), "Prompt must be a string for video generation"
                gen_video_op: GenerateVideosOperation = await self.api_client.aio.models.generate_videos(
                    model=self.model_name,
                    prompt=prompt,
                    **kwargs
                )
                while not gen_video_op.done:
                    await asyncio.sleep(5)
                    gen_video_op = await self.api_client.aio.operations.get(gen_video_op)
                    if gen_video_op.error:
                        raise RuntimeError(f"Video generation failed: {gen_video_op.error}")
                return gen_video_op.result.generated_videos

            elif modality == "EMBEDDINGS":
                response: EmbedContentResponse = await self.api_client.aio.models.embed_content(
                    model=self.model_name,
                    contents=prompt,
                    **kwargs
                )
                return response

            else:  # Default is TEXT
                config_data = kwargs.pop("config", {})

                if not isinstance(config_data, GenerateContentConfig):
                    config = GenerateContentConfig(**config_data)
                else:
                    config = config_data

                if self.system_instruction and not config.system_instruction:
                    config.system_instruction = self.system_instruction

                logger.info(config)
                response: GenerateContentResponse = await self.api_client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
                return response.text if hasattr(response, "text") else response

        except Exception as e:
            logger.error(f"[GeminiClient] Error generating content: {e}")
            raise

    @classmethod
    async def list_models(cls, *args: Any, **kwargs: Any) -> Sequence[LLMSchema]:
        """List Gemini models available to the configured API key.

        Args:
            args: Positional arguments forwarded to `models.list`.
            kwargs: Keyword arguments forwarded to `models.list`.

        Returns:
            Sequence of Gemini model metadata.
        """
        try:
            api_client = cls.get_api_client()
            available_models = await api_client.aio.models.list(*args, **kwargs)

            return [
                LLMSchema(
                    name=model_info.name,
                    display_name=model_info.display_name,
                    description=model_info.description,
                    client="google",
                )
                for model_info in available_models
                if hasattr(model_info, "supported_actions") and (
                    "generateContent" in model_info.supported_actions or
                    "embedContent" in model_info.supported_actions
                )
            ]
        except Exception as e:
            logger.error(f"[GeminiClient] Failed to list models: {e}")
            return []


# === Example usage ===
if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    async def main() -> None:
        """Run a small manual Gemini smoke test."""
        print("Available Gemini Models:")
        for model in await GeminiClient.list_models():
            print(f"- {model.name}")

        try:
            client = GeminiClient.get_model(
                "models/gemini-2.0-flash-exp",
                system_instruction="generate the output in Spanish"
            )
            response = await client.generate("Tell me a joke about AI.")
            print("\nGemini Response:\n")
            print(response)
        except Exception as e:
            logger.error(f"Error during generation: {e}")

    asyncio.run(main())
