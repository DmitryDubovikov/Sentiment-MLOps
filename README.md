# Sentiment MLOps

End-to-end MLOps pipeline for sentiment classification using scikit-learn, MLflow, and Docker.

## Overview

This project implements a production-like ML workflow for binary sentiment analysis (positive/negative reviews). It demonstrates experiment tracking, model versioning, artifact storage, and containerized training.

### Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | scikit-learn (LogisticRegression + TF-IDF) |
| Experiment Tracking | MLflow 3.x |
| Artifact Storage | MinIO (S3-compatible) |
| Backend Store | PostgreSQL |
| Package Manager | uv |
| Containerization | Docker Compose |

## Quick Start

### Prerequisites

- Docker & Docker Compose

### Start Services

```bash
docker compose up -d --build
```

This starts:
- **PostgreSQL** (port 5432) — MLflow metadata store
- **MinIO** (ports 9000, 9001) — artifact storage
- **MLflow** (port 5001) — tracking server & UI
- **Training** — container for running pipelines

### Train Model

```bash
docker compose exec training uv run python pipelines/train_simple.py
```

### Test Predictions

```bash
docker compose exec training uv run python pipelines/test_model.py
```

With custom text:
```bash
docker compose exec training uv run python pipelines/test_model.py --text 'Great movie!'
```

## Web Interfaces

| Service | URL | Credentials |
|---------|-----|-------------|
| MLflow UI | http://localhost:5001 | — |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |

### Exploring MLflow UI

After running training, open http://localhost:5001:

1. **Experiments** (left sidebar) — select `sentiment-classifier` to see all training runs
2. **Runs table** — each row is a training run with metrics, parameters, and timestamps
3. **Run details** (click on a run):
   - **Parameters** tab: hyperparameters (C, max_iter, vectorizer settings)
   - **Metrics** tab: accuracy, F1, precision, recall with history charts
   - **Artifacts** tab: saved model files (model.pkl, requirements.txt, MLmodel)
4. **Compare runs** — select multiple runs with checkboxes, click "Compare" to see metrics side-by-side
5. **Model info** — click on the model artifact to see input/output schema and example predictions

### Exploring MinIO Console

Open http://localhost:9001 and login with `minioadmin` / `minioadmin`:

1. **Object Browser** (left menu) — view all buckets:
   - `mlflow-artifacts` — MLflow experiment artifacts and models
   - `models` — production models (used in later iterations)
   - `data` — DVC remote storage (used in later iterations)
2. **Inside mlflow-artifacts**:
   - Navigate to `1/models/<model-id>/artifacts/` to see saved model files
   - `model.pkl` — serialized sklearn pipeline (~200KB)
   - `MLmodel` — MLflow model metadata (YAML)
   - `requirements.txt` — Python dependencies for model serving
   - `input_example.json` — sample input for testing
3. **Bucket settings** — click bucket name → Settings to see versioning, lifecycle rules

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
docker compose down -v     # Stop and remove volumes
```
