.PHONY: deploy publish docs docs-check

docs-check:
	uv run --group docs mkdocs build --strict -d /tmp/modihub-mkdocs-check

docs:
	uv run --group docs mkdocs build --strict

publish:
	@echo "Building and publishing package..."
	@export $(shell grep -v '^#' .env | xargs) && \
	uv build && \
	uv publish --token $$PYPI_TOKEN
