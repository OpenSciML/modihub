import itertools
from collections.abc import Iterator, Sequence
from typing import Any

from .openai_client import OpenAIClient
from .gemini_client import GeminiClient
from .ollama_client import OllamaClient
from .groq_client import GroqClient
from .anthropic_client import AnthropicClient
from .base import LLMClient, LLMSchema

class ModelsList:
    """Container for model metadata with grouping and filtering helpers.

    Args:
        models: Model metadata records to expose through list-like operations.

    Attributes:
        models: Ordered model metadata records.
    """

    def __init__(self, models: Sequence[LLMSchema]) -> None:
        """Initialize the container.

        Args:
            models: Model metadata records to expose through list-like operations.
        """
        self.models = models

    def __iter__(self) -> Iterator[LLMSchema]:
        """Iterate over model metadata records.

        Returns:
            Iterator over model metadata records.
        """
        return iter(self.models)

    def __len__(self) -> int:
        """Return the number of models.

        Returns:
            Count of model metadata records.
        """
        return len(self.models)

    def __getitem__(self, item: int | slice) -> LLMSchema | Sequence[LLMSchema]:
        """Return a model or model slice by index.

        Args:
            item: Integer index or slice.

        Returns:
            Selected model metadata or model metadata sequence.
        """
        return self.models[item]

    def group_by(self, key: str) -> Iterator[tuple[Any, list[LLMSchema]]]:
        """Group models by an attribute value.

        Args:
            key: Model metadata attribute to group by.

        Returns:
            Iterator of attribute values and grouped model metadata.
        """
        sorted_models = sorted(self.models, key=lambda m: getattr(m, key))
        for group_key, group_items in itertools.groupby(sorted_models, lambda m: getattr(m, key)):
            yield group_key, list(group_items)

    def filter_by(self, key: str, value: str) -> "ModelsList":
        """Filter models by an attribute value.

        Args:
            key: Model metadata attribute to inspect.
            value: Required attribute value.

        Returns:
            ModelsList containing matching records.
        """
        return ModelsList([model for model in self.models if getattr(model, key) == value])

    def __repr__(self) -> str:
        """Return a readable provider-prefixed model list.

        Returns:
            Newline-separated provider and model names.
        """
        return "\n".join([f"{model.client}: {model.name}" for model in self.models])

class LLM:
    """Factory for asynchronous model clients."""

    _clients = {"openai": OpenAIClient, "google": GeminiClient, "ollama": OllamaClient, "groq": GroqClient, "anthropic": AnthropicClient}

    @staticmethod
    async def available_models() -> ModelsList:
        """Get models available from all configured clients.

        Returns:
            ModelsList containing metadata from providers that could be queried.
        """
        models = []
        for client_name, client_class in LLM._clients.items():
            try:
                models.extend(await client_class.list_models())
            except Exception:
                pass
        return ModelsList(models)

    @staticmethod
    async def create(model: str, **kwargs: Any) -> LLMClient:
        """Create an asynchronous model client for a configured provider.

        Args:
            model: Provider model name to instantiate.
            kwargs: Additional keyword arguments passed to the client constructor.

        Returns:
            Initialized asynchronous model client.

        Raises:
            ValueError: If the model is not found in any registered provider.
        """
        for client_name, client_class in LLM._clients.items():
            try:
                available_model_names = {m.name for m in await client_class.list_models()}
                if model in available_model_names:
                    return client_class.get_model(model, **kwargs)
            except Exception:
                continue
        raise ValueError(f"Model '{model}' is not available in any registered client.")
