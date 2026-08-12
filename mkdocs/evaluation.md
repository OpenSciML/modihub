# Evaluation

The `Evaluator` helper runs multiple model clients concurrently and scores each response with configured metrics.

## Basic Evaluation

```python
import asyncio

from dotenv import find_dotenv, load_dotenv
from modihub.eval import Evaluator
from modihub.metrics import LexicalDiversity, Perplexity


async def main() -> None:
    load_dotenv(find_dotenv())

    evaluator = Evaluator(
        models=[
            "gpt-4o-mini",
            "models/gemini-1.5-flash-latest",
        ],
        metrics=[
            Perplexity(),
            LexicalDiversity(),
        ],
    )

    results = await evaluator.evaluate("Explain asynchronous programming.")
    for model, scores in zip(evaluator.models, results):
        print(model, scores)


asyncio.run(main())
```

## Included Metrics

`Perplexity` is a lightweight heuristic based on average token length and repetition. It is not a language-model perplexity calculation.

`LexicalDiversity` reports the fraction of unique tokens in the generated output.

Both metrics use NLTK tokenization, so applications may need to ensure the relevant NLTK tokenizer data is available in their runtime environment.
