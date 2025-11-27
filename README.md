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

### Step 1: Start Services

```bash
# Clean start (removes previous data)
docker compose down -v

# Build and start all services
docker compose up -d --build

# Verify all services are running
docker compose ps
```

**Expected:** 6 services running (postgres, minio, mlflow, prefect-server, training, fastapi)

### Step 2: Verify Web UIs

| Service | URL | What to see |
|---------|-----|-------------|
| Prefect UI | http://localhost:4200 | Dashboard with empty Flow Runs |
| MLflow UI | http://localhost:5001 | Experiments page |
| MinIO Console | http://localhost:9001 | Login: `minioadmin` / `minioadmin` |
| FastAPI Docs | http://localhost:8000/docs | Swagger UI (API unhealthy until champion model exists) |

### Step 3: Run First Training

```bash
docker compose exec training uv run python -m pipelines.cli train
```

**Expected output:**
```
Starting training flow with data: data/imdb_sample.csv
Metrics: {'accuracy': 0.835, 'f1': 0.839, ...}
MLflow run ID: <run_id>
Registered model version: 1

==================================================
Training completed!
==================================================
Run ID: <run_id>
Metrics:
  accuracy: 0.8350
  f1: 0.8390
  precision: 0.8190
  recall: 0.8600
Model version: 1
```

### Step 4: Explore Prefect UI

Open http://localhost:4200:

1. Go to **Flow Runs** in left menu
2. Find `sentiment-training` run with status **Completed** (green)
3. Click on the run to see:
   - Task timeline: `load-data` → `split-data` → `train-model` → `evaluate-model` → `log-mlflow` → `register-model`
   - Logs with training output
   - Duration of each task

### Step 5: Explore MLflow Model Registry

Open http://localhost:5001:

1. Go to **Models** in left menu (not Experiments!)
2. Find `sentiment-classifier` model
3. Click to see Version 1 registered
4. Note: **Aliases** column is empty (no champion yet)

### Step 6: Train with Champion Promotion

```bash
docker compose exec training uv run python -m pipelines.cli train --champion
```

**Expected output:**
```
...
Model version: 2
Status: Set as champion
```

### Step 7: Verify Champion in MLflow

Refresh http://localhost:5001 → Models → sentiment-classifier:

- Version 2 now has alias **champion**
- Click on Version 2 to see alias details

### Step 8: Test FastAPI Inference API

Now that a champion model exists, the API is ready:

```bash
# Check API health
curl http://localhost:8000/health
# Expected: {"status":"healthy","model_loaded":true}

# Get sentiment prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie is amazing!"}'
# Expected: {"sentiment":"positive","confidence":0.73,"model_version":"2"}

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Terrible film, waste of time"}'
# Expected: {"sentiment":"negative","confidence":0.65,"model_version":"2"}

# Get current model info
curl http://localhost:8000/model-info
# Expected: {"model_name":"sentiment-classifier","version":"2","alias":"champion",...}
```

Or use Swagger UI at http://localhost:8000/docs

### Step 9: Use CLI Commands

```bash
# Show champion model info
docker compose exec training uv run python -m pipelines.cli model-info

# List all model versions with aliases
docker compose exec training uv run python -m pipelines.cli list-versions

# Manually set alias on a version
docker compose exec training uv run python -m pipelines.cli set-alias \
    --model-name sentiment-classifier --version 1 --alias challenger
```

### Step 10: Test Champion Comparison & Auto-Reload

Run training again with `--champion`:

```bash
docker compose exec training uv run python -m pipelines.cli train --champion
```

**Expected behavior:**
- If new model is **better** → becomes new champion
- If new model is **worse or equal** → `Status: Not promoted (current champion is better)`

Check MLflow UI to verify which version has the `champion` alias.

**API auto-reload:** FastAPI checks for new champion models every 60 seconds. After promoting a new champion, the API will automatically load it. You can also force reload:

```bash
curl -X POST http://localhost:8000/admin/reload
```

### Step 11: View All Flow Runs in Prefect

Open http://localhost:4200 → Flow Runs:

- See all completed training runs
- Compare execution times
- Review task-level details

## CLI Reference

| Command | Description |
|---------|-------------|
| `train` | Run training pipeline, register model |
| `train --champion` | Train and promote to champion if better |
| `model-info` | Show current champion model details |
| `list-versions` | List all model versions with aliases |
| `set-alias` | Manually set alias on model version |

## API Reference

FastAPI inference service available at http://localhost:8000

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (model loaded status) |
| `/predict` | POST | Get sentiment prediction |
| `/model-info` | GET | Current model version and metrics |
| `/admin/reload` | POST | Force model reload from registry |
| `/docs` | GET | Swagger UI |

**Example requests:**

```bash
# Predict sentiment
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great product, highly recommend!"}'

# Response: {"sentiment":"positive","confidence":0.82,"model_version":"2"}
```

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
2. **Runs table** — each row is a training run with metrics and timestamps
3. **Run details** (click on a run):
   - **Parameters** tab: hyperparameters (C, max_iter, vectorizer settings)
   - **Metrics** tab: accuracy, F1, precision, recall
   - **Artifacts** tab: saved model files
4. **Compare runs** — select multiple runs, click "Compare"

**Models view** (http://localhost:5001/#/models):

1. **Registered Models** — list of all registered models
2. **Model versions** — click model to see all versions
3. **Aliases** — see which version is `champion`
4. **Version details** — source run, creation time, artifacts

### Exploring MinIO Console

Open http://localhost:9001 (login: `minioadmin` / `minioadmin`):

1. **Object Browser** — view buckets:
   - `mlflow-artifacts` — MLflow experiment artifacts and models
   - `models` — production models
   - `data` — DVC remote storage
2. **Inside mlflow-artifacts**:
   - Navigate to see saved model files
   - `model.pkl` — serialized sklearn pipeline
   - `MLmodel` — MLflow model metadata
   - `requirements.txt` — Python dependencies

### Exploring Prefect UI

Open http://localhost:4200:

1. **Flow Runs** — list of all pipeline executions
2. **Run details** — click on a run to see:
   - Task dependency graph
   - Task execution timeline
   - Logs and outputs
3. **Flows** — registered flow definitions

### FastAPI Service

Open http://localhost:8000/docs for interactive Swagger UI:

1. **Try endpoints** — test predictions directly in browser
2. **Schema** — view request/response models
3. **Auto-reload** — model updates automatically when new champion is promoted (every 60s)

**Note:** API returns 503 until a champion model exists. Run training with `--champion` first.

## Model Registry Strategy

This project uses **MLflow Aliases** for model lifecycle management:

| Alias | Purpose |
|-------|---------|
| `@champion` | Production model for serving |
| `@challenger` | Candidate for A/B testing (optional) |

## Configuration

Environment variables (see `.env.example`):

```bash
MLFLOW_TRACKING_URI=http://localhost:5001
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
```

## Stopping Services

```bash
docker compose down        # Stop containers
docker compose down -v     # Stop and remove volumes (clean reset)
```

## Development Setup

### Local Development

```bash
# Install uv package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --dev

# Run linting
uv run ruff check src/ app/ pipelines/
uv run ruff format --check src/ app/ pipelines/

# Run tests
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ --cov=src --cov=app --cov=pipelines --cov-report=term-missing
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run all hooks manually
uv run pre-commit run --all-files
```

Pre-commit hooks include:
- **ruff**: Python linting and formatting
- **trailing-whitespace**: Remove trailing whitespace
- **end-of-file-fixer**: Ensure files end with newline
- **check-yaml**: Validate YAML syntax
- **detect-secrets**: Scan for accidentally committed secrets
- **hadolint**: Dockerfile linting
- **commitizen**: Commit message validation

## Data Version Control (DVC)

The project uses DVC for data versioning with MinIO as remote storage.

### DVC Setup

```bash
# Initialize DVC (already done)
dvc init

# Configure MinIO remote (already configured in .dvc/config)
dvc remote add -d minio s3://data
dvc remote modify minio endpointurl http://localhost:9000
```

### DVC Commands

```bash
# Prepare dataset
uv run python scripts/prepare_data.py --output data/imdb_sample.csv

# Push data to remote
dvc push

# Pull data from remote
dvc pull

# Run DVC pipeline (prepare_data + train)
dvc repro
```

### DVC Pipeline

The `dvc.yaml` defines a reproducible pipeline:

1. **prepare_data**: Downloads IMDb dataset and creates CSV
2. **train**: Runs training with MLflow tracking

```bash
# View pipeline DAG
dvc dag

# Run specific stage
dvc repro prepare_data
dvc repro train
```

## CI/CD

The project uses GitHub Actions for continuous integration and deployment.

### Workflows

| Workflow | Trigger | Description |
|----------|---------|-------------|
| `ci.yml` | Push/PR | Lint, test, build images |
| `train.yml` | Manual | Run training pipeline |
| `release.yml` | Tag | Build & push to GHCR, create release |

### CI Pipeline

On every push and pull request:
1. **Lint**: Run ruff linter and formatter check
2. **Test**: Run pytest with coverage
3. **Build**: Build Docker images (on main branch)

### Manual Training

Trigger training via GitHub Actions:
1. Go to Actions → "Train Model"
2. Click "Run workflow"
3. Configure samples count and champion promotion
4. View training metrics in artifacts

### Release Process

```bash
# Create version tag
git tag v1.0.0
git push origin v1.0.0

# GitHub Actions will:
# - Build and push images to ghcr.io
# - Create GitHub release with notes
```

## Testing

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures
├── unit/
│   ├── test_data.py      # Data loading tests
│   ├── test_model.py     # Model training tests
│   └── test_preprocessing.py  # Text preprocessing tests
└── integration/
    └── test_api.py       # FastAPI endpoint tests
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run unit tests only
uv run pytest tests/unit/ -v

# Run integration tests only
uv run pytest tests/integration/ -v

# Run with coverage
uv run pytest tests/ --cov=src --cov=app --cov-report=html
```

## Production Deployment

### Production Configuration

Use `docker-compose.prod.yml` for production deployments:

```bash
# Set required environment variables
export POSTGRES_USER=your_secure_user
export POSTGRES_PASSWORD=your_secure_password
export MINIO_ROOT_USER=your_minio_user
export MINIO_ROOT_PASSWORD=your_minio_password

# Start production stack
docker compose -f docker-compose.prod.yml up -d
```

### Production Features

- Resource limits on all containers
- Health checks with stricter intervals
- JSON logging for log aggregation
- Nginx reverse proxy (optional profile)
- No hot-reload (5-minute model check interval)
- SSL/TLS ready configuration

### With Nginx Reverse Proxy

```bash
# Start with nginx profile
docker compose -f docker-compose.prod.yml --profile with-nginx up -d

# Access services via nginx:
# - API: http://localhost/api/
# - MLflow: http://localhost/mlflow/
# - MinIO: http://localhost/minio/
```

### SSL Configuration

For production SSL:

1. Place certificates in `docker/nginx/ssl/`:
   - `cert.pem`: SSL certificate
   - `key.pem`: Private key

2. Uncomment HTTPS server block in `docker/nginx/nginx.conf`

3. Enable HTTPS redirect in HTTP server block

### Environment Variables

Production environment requires these variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `POSTGRES_USER` | PostgreSQL username | Yes |
| `POSTGRES_PASSWORD` | PostgreSQL password | Yes |
| `MINIO_ROOT_USER` | MinIO access key | Yes |
| `MINIO_ROOT_PASSWORD` | MinIO secret key | Yes |

## Security

### Dependabot

Automated dependency updates are configured for:
- Python packages (weekly)
- Docker base images (weekly)
- GitHub Actions (weekly)

### Secrets Scanning

Pre-commit hooks include `detect-secrets` to prevent accidental secret commits.

Update baseline after intentional changes:
```bash
detect-secrets scan --baseline .secrets.baseline
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
└── docker-compose.yml     # Local development stack
```
