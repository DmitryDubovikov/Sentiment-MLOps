# Sentiment MLOps

End-to-end MLOps pipeline for sentiment classification using scikit-learn, MLflow, Prefect, FastAPI, and Docker.

## Overview

This project implements a production-like ML workflow for binary sentiment analysis (positive/negative reviews). It demonstrates experiment tracking, model versioning, artifact storage, orchestration, model serving, and containerized deployment.

### Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | scikit-learn (LogisticRegression + TF-IDF) |
| Model Serving | FastAPI + uvicorn |
| Experiment Tracking | MLflow 3.x |
| Orchestration | Prefect 3.x |
| Model Registry | MLflow Model Registry with Aliases |
| Artifact Storage | MinIO (S3-compatible) |
| Backend Store | PostgreSQL |
| Data Versioning | DVC with S3 remote |
| CI/CD | GitHub Actions |
| Package Manager | uv |
| Containerization | Docker Compose |
| Code Quality | ruff, pre-commit |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Make (usually pre-installed)

### View Available Commands

```bash
make help
```

### One-Command Setup

```bash
make quickstart
```

This will:
1. Start all services (MLflow, MinIO, Prefect, API)
2. Build CLI container
3. Prepare data via DVC
4. Train model and set as champion

### Manual Setup

#### Step 1: Start Services

```bash
make up
```

**Expected:** Services started with URLs:
- MLflow: http://localhost:5001
- MinIO: http://localhost:9001
- Prefect: http://localhost:4200
- API Docs: http://localhost:8000/docs

#### Step 2: Verify Services

```bash
make status
```

#### Step 3: Build CLI Container

```bash
make build-cli
```

#### Step 4: Run First Training

```bash
make train-champion
```

**Expected output:**
```
Starting training flow with data: data/imdb_sample.csv
Metrics: {'accuracy': 0.835, 'f1': 0.839, ...}
MLflow run ID: <run_id>
Registered model version: 1
Model set as champion

Training completed!
```

#### Step 5: Test Prediction

```bash
make predict TEXT="This movie is amazing!"
```

**Expected:**
```json
{
    "sentiment": "positive",
    "confidence": 0.82,
    "probabilities": {"negative": 0.18, "positive": 0.82}
}
```

## Makefile Commands

### Infrastructure

| Command | Description |
|---------|-------------|
| `make up` | Start all services |
| `make down` | Stop all services |
| `make restart` | Restart all services |
| `make logs` | Show logs for all services |
| `make status` | Show status of all services |

### Training

| Command | Description |
|---------|-------------|
| `make train` | Train model with Prefect |
| `make train-champion` | Train and set as champion |
| `make train-simple` | Train without Prefect |

### Testing

| Command | Description |
|---------|-------------|
| `make test` | Run all tests |
| `make test-unit` | Run unit tests only |
| `make test-integration` | Run integration tests only |
| `make test-cov` | Run tests with coverage |

### Code Quality

| Command | Description |
|---------|-------------|
| `make lint` | Run ruff linter |
| `make format` | Format code with ruff |
| `make check` | Run lint + format check |

### Data (DVC)

| Command | Description |
|---------|-------------|
| `make dvc-repro` | Reproduce DVC pipeline |
| `make dvc-push` | Push data to remote |
| `make dvc-pull` | Pull data from remote |

### Utilities

| Command | Description |
|---------|-------------|
| `make models` | List registered models |
| `make predict TEXT="..."` | Make prediction |
| `make shell` | Open shell in CLI container |
| `make clean` | Remove containers and volumes |
| `make quickstart` | Full setup in one command |

## Web Interfaces

| Service | URL | Credentials |
|---------|-----|-------------|
| FastAPI Docs | http://localhost:8000/docs | — |
| MLflow UI | http://localhost:5001 | — |
| Prefect UI | http://localhost:4200 | — |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |

### Exploring MLflow UI

**Experiments view** (http://localhost:5001):

1. **Experiments** (left sidebar) — select `sentiment-classifier`
2. **Runs table** — each row is a training run with metrics
3. **Run details** (click on a run):
   - **Parameters** tab: hyperparameters
   - **Metrics** tab: accuracy, F1, precision, recall
   - **Artifacts** tab: saved model files

**Models view** (http://localhost:5001/#/models):

1. **Registered Models** — list of all registered models
2. **Model versions** — click model to see all versions
3. **Aliases** — see which version is `champion`

### Exploring Prefect UI

Open http://localhost:4200:

1. **Flow Runs** — list of all pipeline executions
2. **Run details** — task dependency graph, timeline, logs

### Exploring MinIO Console

Open http://localhost:9001 (login: `minioadmin` / `minioadmin`):

1. **Object Browser** — view buckets:
   - `mlflow-artifacts` — MLflow experiment artifacts
   - `dvc` — DVC remote storage

## API Reference

FastAPI inference service at http://localhost:8000

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Get sentiment prediction |
| `/model/info` | GET | Current model info |
| `/model/reload` | POST | Force model reload |
| `/docs` | GET | Swagger UI |

**Example:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great product!"}'
```

## Model Registry Strategy

This project uses **MLflow Aliases** for model lifecycle:

| Alias | Purpose |
|-------|---------|
| `@champion` | Production model for serving |
| `@challenger` | Candidate for A/B testing (optional) |

## CI/CD

### Workflows

| Workflow | Trigger | Description |
|----------|---------|-------------|
| `ci.yml` | Push/PR | Lint, test, build images |
| `train.yml` | Manual | Run training pipeline |
| `release.yml` | Tag | Build & push to GHCR |

### Release Process

```bash
git tag v1.0.0
git push origin v1.0.0
```

GitHub Actions will build and push images to ghcr.io.

## Testing

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures
├── unit/
│   ├── test_data.py      # Data loading tests
│   ├── test_model.py     # Model training tests
│   └── test_preprocessing.py  # Preprocessing tests
└── integration/
    └── test_api.py       # FastAPI endpoint tests
```

### Running Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests only
make test-integration

# With coverage
make test-cov
```

## Production Deployment

### Production Stack

```bash
docker compose -f docker-compose.prod.yml up -d
```

### Production Features

- Resource limits on all containers
- Health checks with stricter intervals
- JSON logging for log aggregation
- Nginx reverse proxy
- No hot-reload (5-minute model check interval)

### With Nginx

```bash
docker compose -f docker-compose.prod.yml --profile with-nginx up -d
```

## Configuration

Environment variables (see `.env.example`):

```bash
MLFLOW_TRACKING_URI=http://localhost:5001
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
```

## Project Structure

```
.
├── app/                    # FastAPI serving application
│   ├── main.py            # API endpoints
│   ├── model_loader.py    # Model loading from MLflow
│   └── schemas.py         # Pydantic models
├── src/                    # Core ML logic
│   ├── data.py            # Data loading utilities
│   ├── model.py           # Model training
│   └── preprocessing.py   # Text preprocessing
├── pipelines/             # Prefect orchestration
│   ├── train.py           # Training flow
│   ├── cli.py             # CLI commands
│   └── tasks/             # Prefect tasks
├── tests/                 # Test suite
├── scripts/               # Utility scripts
├── docker/                # Docker configurations
├── .github/               # GitHub Actions & Dependabot
├── dvc.yaml               # DVC pipeline definition
├── params.yaml            # Training parameters
├── Makefile               # All commands
└── docker-compose.yml     # Development stack
```

## Detailed Testing Guide

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for step-by-step testing instructions with expected outputs.
