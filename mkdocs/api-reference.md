# API Reference

This page summarizes the public API exposed by the current package.

## `modihub.llm.LLM`

Factory class for provider-backed asynchronous clients.

### `await LLM.available_models()`

Returns a `ModelsList` containing `LLMSchema` records from all providers that can be queried with the current environment.

### `await LLM.create(model: str, **kwargs)`

Returns an `LLMClient` for the requested model. Additional keyword arguments are passed to the provider client constructor.

## `modihub.llm.ModelsList`

Container returned by model discovery.

- `len(models)` returns the number of records.
- `models[index]` returns a model record.
- `models.group_by("client")` yields provider names and matching model records.
- `models.filter_by("client", "openai")` returns another `ModelsList`.

## `modihub.llm.LLMSchema`

Pydantic model describing provider model metadata.

| Field | Type | Description |
| --- | --- | --- |
| `client` | `str` | Provider name. |
| `name` | `str` | Provider model identifier. |
| `display_name` | `str` | Human-readable model name. |
| `description` | `str \| None` | Optional provider description. |

## `modihub.llm.LLMClient`

Abstract base class for provider clients.

### `await client.generate(prompt, **kwargs)`

Generates a response using the provider model.

### `await client(prompt, **kwargs)`

Convenience wrapper around `generate`.

## `modihub.eval.Evaluator`

Runs multiple model calls concurrently and applies metric instances to each output.

### `await evaluator.evaluate(prompt: str)`

Returns a list of metric score dictionaries in the same order as `evaluator.models`.

## `modihub.metrics`

`Metric` is the abstract base class for scoring generated text.

Included metric implementations:

- `Perplexity`
- `LexicalDiversity`
