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
| Package Manager | uv |
| Containerization | Docker Compose |

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
