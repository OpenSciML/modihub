from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, Field

ClientT = TypeVar("ClientT", bound="LLMClient")


class LLMSchema(BaseModel):
    """Metadata describing a model exposed by a provider client."""

    client: str = Field(
        ..., title="Client Name", description="The name of the client that provides the model"
    )
    name: str = Field(
        ..., title="Model Name", description="The name of the model to use"
    )
    display_name: str = Field(
        ..., title="Display Name", description="The display name of the model"
    )
    description: str | None = Field(
        default=None, title="Model Description", description="A description of the model"
    )



class LLMClient(ABC):
    """Base class for asynchronous LLM clients.

    Args:
        model_name: Provider-specific model identifier used for generation calls.
        args: Positional arguments reserved for provider subclasses.
        kwargs: Keyword arguments reserved for provider subclasses.

    Attributes:
        model_name: Provider-specific model identifier used by the client.
    """

    def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the client with the selected provider model.

        Args:
            model_name: Provider-specific model identifier used for generation calls.
            args: Positional arguments accepted by subclass constructors.
            kwargs: Keyword arguments accepted by subclass constructors.
        """
        self.model_name = model_name

    def __new__(cls, *args: Any, **kwargs: Any) -> LLMClient:
        """Prevent direct construction outside the factory helpers.

        Args:
            args: Positional constructor arguments.
            kwargs: Keyword constructor arguments.

        Returns:
            A newly allocated client instance when called through subclasses.

        Raises:
            TypeError: If callers instantiate a client class directly.
        """
        raise TypeError("Direct instantiation is not allowed. Use 'get_model' method.")

    @classmethod
    def get_model(cls: type[ClientT], *args: Any, **kwargs: Any) -> ClientT:
        """Create a model instance without making network validation calls.

        Args:
            args: Positional constructor arguments passed to the subclass.
            kwargs: Keyword constructor arguments passed to the subclass.

        Returns:
            Initialized provider client.

        Raises:
            TypeError: If called on a class that is not an `LLMClient` subclass.
        """
        if not issubclass(cls, LLMClient):
            raise TypeError("Method can only be called on subclasses of LLM")
        instance = super().__new__(cls)
        instance.__init__(*args, **kwargs)
        return instance

    @staticmethod
    @abstractmethod
    async def list_models() -> Sequence[LLMSchema]:
        """List models available from the provider.

        Returns:
            Sequence of model metadata exposed by the provider.
        """
        raise NotImplementedError

    @abstractmethod
    async def generate(self, prompt: Any, *args: Any, **kwargs: Any) -> str:
        """Generate a model response for a prompt.

        Args:
            prompt: Provider-supported prompt payload.
            args: Additional positional generation arguments.
            kwargs: Additional provider-specific generation options.

        Returns:
            Generated text response.
        """
        raise NotImplementedError

    async def __call__(self, prompt: Any, *args: Any, **kwargs: Any) -> str:
        """Generate a response when the client is called directly.

        Args:
            prompt: Provider-supported prompt payload.
            args: Additional positional generation arguments.
            kwargs: Additional provider-specific generation options.

        Returns:
            Generated text response.
        """
        return await self.generate(prompt, *args, **kwargs)
