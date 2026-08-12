# Development

## Install The Repository

```bash
uv sync
```

## Validate Python Sources

The repository currently uses syntax/import smoke checks rather than a full test suite.

```bash
uv run python -m py_compile \
  src/modihub/__init__.py \
  src/modihub/eval/evaluator.py \
  src/modihub/metrics/metrics.py \
  src/modihub/utils/image_utils.py \
  src/modihub/llm/base.py \
  src/modihub/llm/openai_client.py \
  src/modihub/llm/anthropic_client.py \
  src/modihub/llm/gemini_client.py \
  src/modihub/llm/groq_client.py \
  src/modihub/llm/ollama_client.py \
  src/modihub/llm/llm.py
```

## Build The Package

```bash
uv build
```

## Build The Documentation

Run a temporary strict docs build:

```bash
make docs-check
```

Generate the GitHub Pages-ready site into `docs/`:

```bash
make docs
```

Editable documentation source files live in `mkdocs/`. The generated HTML output lives in `docs/`.
