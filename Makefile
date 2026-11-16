.DEFAULT_GOAL := help
.PHONY: help install fmt lint test web web-dev up down migrate revision

help: ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install: ## Create the virtualenv and install dependencies
	uv sync

fmt: ## Format and auto-fix
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Formatting, lint and type checks (what CI runs)
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy

test: ## Run tests; integration tests need INTELLIQUERY_DATABASE_URL
	uv run pytest --cov

web: ## Type-check, test and build the web UI into web/dist
	cd web && npm ci && npm run typecheck && npm test && npm run build

web-dev: ## Run the web UI with hot reload (expects the API on port 8000)
	cd web && npm run dev

up: ## Start the full stack
	docker compose up --build -d

down: ## Stop the stack (data volume is kept)
	docker compose down

migrate: ## Apply migrations to the configured database
	uv run alembic upgrade head

revision: ## Autogenerate a migration: make revision m="add widgets"
	uv run alembic revision --autogenerate -m "$(m)"
