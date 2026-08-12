import asyncio

from modihub.llm import LLM, LLMClient
from modihub.metrics import Metric

class Evaluator:
    """Benchmark multiple asynchronous model clients with text metrics.

    Args:
        models: Provider model names to evaluate.
        metrics: Metric instances used to score each model output.

    Attributes:
        models: Provider model names to evaluate.
        metrics: Metric instances used to score each model output.
    """

    def __init__(self, models: list[str], metrics: list[Metric]) -> None:
        """Initialize the evaluator.

        Args:
            models: Provider model names to evaluate.
            metrics: Metric instances used to score each model output.
        """
        self.models = models
        self.metrics = metrics

    async def evaluate(self, prompt: str) -> list[dict[str, float]]:
        """Evaluate all configured models on a prompt.

        Args:
            prompt: Text prompt to send to each model.

        Returns:
            List of metric score dictionaries in the same order as `self.models`.
        """
        clients = await asyncio.gather(*(LLM.create(model) for model in self.models))
        return await asyncio.gather(
            *(self._evaluate_model(client, prompt) for client in clients)
        )

    async def _evaluate_model(self, model: LLMClient, prompt: str) -> dict[str, float]:
        """Evaluate a single model output using all configured metrics.

        Args:
            model: Asynchronous model client to evaluate.
            prompt: Text prompt for the model to answer.

        Returns:
            Mapping from metric class name to score.
        """
        output = await model(prompt)
        return {
            metric.__class__.__name__: metric(output)
            for metric in self.metrics
        }
