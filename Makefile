# Sentiment MLOps - Makefile
# All commands run through Docker - no local Python/uv required

.PHONY: help up down restart logs status \
        build build-cli build-api \
        train train-champion train-simple \
        test test-unit test-integration test-cov \
        lint format check \
        dvc-repro dvc-push dvc-pull \
        models predict shell clean

# Default target
help:
	@echo "Sentiment MLOps - Available Commands"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make up              Start all services (mlflow, minio, prefect, api)"
	@echo "  make down            Stop all services"
	@echo "  make restart         Restart all services"
	@echo "  make logs            Show logs for all services"
	@echo "  make status          Show status of all services"
	@echo ""
	@echo "Build:"
	@echo "  make build           Build all Docker images"
	@echo "  make build-cli       Build CLI image only"
	@echo "  make build-api       Build API image only"
	@echo ""
	@echo "Training:"
	@echo "  make train           Train model with Prefect"
	@echo "  make train-champion  Train and set as champion"
	@echo "  make train-simple    Train without Prefect (simple script)"
	@echo ""
	@echo "Testing:"
	@echo "  make test            Run all tests"
	@echo "  make test-unit       Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-cov        Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint            Run ruff linter"
	@echo "  make format          Format code with ruff"
	@echo "  make check           Run lint + format check"
	@echo ""
	@echo "Data (DVC):"
	@echo "  make dvc-repro       Reproduce DVC pipeline (prepare data)"
	@echo "  make dvc-push        Push data to remote storage"
	@echo "  make dvc-pull        Pull data from remote storage"
	@echo ""
	@echo "Utilities:"
	@echo "  make models          List registered models"
	@echo "  make predict TEXT='your text'  Make prediction"
	@echo "  make shell           Open shell in CLI container"
	@echo "  make clean           Remove containers and volumes"
	@echo ""
	@echo "Web UIs:"
	@echo "  MLflow:    http://localhost:5001"
	@echo "  MinIO:     http://localhost:9001"
	@echo "  Prefect:   http://localhost:4200"
	@echo "  API Docs:  http://localhost:8000/docs"

# =============================================================================
# Infrastructure
# =============================================================================

up:
	docker compose up -d --remove-orphans
	@echo ""
	@echo "Services started! Web UIs:"
	@echo "  MLflow:    http://localhost:5001"
	@echo "  MinIO:     http://localhost:9001"
	@echo "  Prefect:   http://localhost:4200"
	@echo "  API Docs:  http://localhost:8000/docs"

down:
	docker compose down --remove-orphans

restart:
	docker compose restart

logs:
	docker compose logs -f

status:
	docker compose ps

# =============================================================================
# Build
# =============================================================================

build:
	docker compose build
	docker compose --profile cli build

build-cli:
	docker compose --profile cli build cli

build-api:
	docker compose build api

# =============================================================================
# Training
# =============================================================================

train:
	docker compose run --rm cli uv run python -m pipelines.cli train

train-champion:
	docker compose run --rm cli uv run python -m pipelines.cli train --champion

train-simple:
	docker compose run --rm cli uv run python pipelines/train_simple.py --download

# =============================================================================
# Testing
# =============================================================================

test:
	docker compose run --rm cli uv run pytest tests/ -v

test-unit:
	docker compose run --rm cli uv run pytest tests/unit/ -v

test-integration:
	docker compose run --rm cli uv run pytest tests/integration/ -v

test-cov:
	docker compose run --rm cli uv run pytest tests/ -v --cov=src --cov=app --cov=pipelines --cov-report=term-missing

# =============================================================================
# Code Quality
# =============================================================================

lint:
	docker compose run --rm cli uv run ruff check src/ app/ pipelines/ scripts/

format:
	docker compose run --rm cli uv run ruff format src/ app/ pipelines/ scripts/

check: lint
	docker compose run --rm cli uv run ruff format --check src/ app/ pipelines/ scripts/

# =============================================================================
# DVC
# =============================================================================

dvc-repro:
	docker compose run --rm cli uv run dvc repro

dvc-push:
	docker compose run --rm cli uv run dvc push

dvc-pull:
	docker compose run --rm cli uv run dvc pull

# =============================================================================
# Utilities
# =============================================================================

models:
	docker compose run --rm cli uv run python -m pipelines.cli list-models

# Usage: make predict TEXT="This movie is great!"
predict:
	@curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"text": "$(TEXT)"}' | python3 -m json.tool

shell:
	docker compose run --rm cli bash

clean:
	docker compose down -v
	docker system prune -f
	@echo "Cleaned up containers and volumes"

# =============================================================================
# Quick Start (full workflow)
# =============================================================================

quickstart: up build-cli
	@echo "Waiting for services to be healthy..."
	@sleep 10
	@echo "Preparing data..."
	$(MAKE) dvc-repro
	@echo "Training model..."
	$(MAKE) train-champion
	@echo ""
	@echo "Quick start complete!"
	@echo "Try: make predict TEXT='This movie is amazing!'"
