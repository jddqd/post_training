.DEFAULT_GOAL := help
.PHONY: help install fmt lint typecheck test check bfcl-questions data data-push clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install deps and git hooks
	uv sync
	uv run pre-commit install

fmt: ## Format and auto-fix
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Lint (no changes)
	uv run ruff check .
	uv run ruff format --check .

typecheck: ## Run mypy
	uv run mypy

test: ## Run tests with coverage
	uv run pytest

check: lint typecheck test ## Lint + typecheck + test

bfcl-questions: ## Export BFCL questions to data/bfcl/ (for decontamination)
	uv run --project eval/bfcl python eval/bfcl/export_questions.py

data: bfcl-questions ## Prepare xLAM into data/xlam/{train,val}.jsonl (~1 min)
	uv run python scripts/prepare_data.py --config configs/data/xlam.yaml

data-push: bfcl-questions ## Prepare xLAM and publish it to the HF repo (commit the code first)
	uv run python scripts/prepare_data.py --config configs/data/xlam.yaml --push

clean: ## Remove caches and build artifacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov build dist
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
