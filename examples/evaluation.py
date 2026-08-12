import asyncio

from dotenv import find_dotenv, load_dotenv
from modihub.metrics import Perplexity, LexicalDiversity
from modihub.eval import Evaluator

async def main() -> None:
    """Evaluate multiple asynchronous model clients across text prompts."""
    load_dotenv(find_dotenv())

    prompts = [
        "What are LLMs?",
        "Explain AI", "What is the meaning of life?"]
    models = [
        "gpt-4o-mini",
        "llama3.1:latest",
        "models/gemini-1.5-flash-latest"
    ]
    metrics = [Perplexity(), LexicalDiversity()]

    evaluator = Evaluator(models, metrics)
    results = {prompt: await evaluator.evaluate(prompt) for prompt in prompts}
    for prompt, result in results.items():
        print(f"Prompt: {prompt}")
        for model, metrics in zip(models, result):
            print(f"{model}: {metrics}")
        print()

asyncio.run(main())
