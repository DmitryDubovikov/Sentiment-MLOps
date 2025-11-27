# Implementation Plan: Sentiment MLOps Project

## Overview

Данный план разбивает реализацию проекта на **4 итерации**. После каждой итерации система будет находиться в рабочем состоянии и может быть протестирована. Итерации сбалансированы по объему работ (~25% каждая).

**Актуальные версии библиотек и образов** (проверено ноябрь 2025):

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.12 | Docker: `python:3.12-slim-bookworm` |
| MLflow | 3.x | Stages deprecated, используем **aliases** |
| Prefect | 3.x | Prefect 3 GA с сентября 2024 |
| FastAPI | 0.115+ | |
| scikit-learn | 1.6+ | Требует Python >=3.10 |
| PostgreSQL | 17 | Docker: `postgres:17` |
| MinIO | latest | `minio/minio:latest` |
| DVC | 3.58+ | |
| uv | 0.9+ | Замена pip/requirements.txt |

---

## Iteration 1: Infrastructure & Basic Training Script ✅

> **Статус**: Завершена. См. `implementation_fact.md` для деталей реализации.

**Ключевые результаты:**
- Docker Compose с postgres, minio, mlflow, training
- MLflow UI: http://localhost:5001 (порт 5000 занят на macOS)
- Внутренний URL: http://mlflow:5000
- Package management через **uv** (не pip)
- Все команды выполняются в контейнере `training`

**Команды:**
```bash
docker compose up -d --build
docker compose exec training uv run python pipelines/train_simple.py
docker compose exec training uv run python pipelines/test_model.py
```

---

## Iteration 2: Prefect Pipeline & Model Registry

**Цель**: Преобразовать training script в reproducible Prefect pipeline. Добавить Model Registry с использованием **aliases**.

### 2.1 Prefect Integration

> **Важно**: Используем Prefect 3.x. Prefect 2.x - legacy.

**Изменения в docker-compose.yml:**

| Service | Image | Ports | Purpose |
|---------|-------|-------|---------|
| `prefect-server` | `prefecthq/prefect:3-python3.12` | 4200 | Prefect UI + API |

**Примечание**: Используем существующий контейнер `training` для запуска flows (не создаём отдельный worker).

```yaml
# Добавить в docker-compose.yml
prefect-server:
  image: prefecthq/prefect:3-python3.12
  container_name: sentiment-prefect
  command: prefect server start --host 0.0.0.0
  ports:
    - "4200:4200"
  environment:
    PREFECT_API_DATABASE_CONNECTION_URL: sqlite+aiosqlite:////root/.prefect/prefect.db
  volumes:
    - prefect_data:/root/.prefect
  networks:
    - mlops-network
```

**Обновить сервис training:**
```yaml
training:
  # ... существующая конфигурация ...
  environment:
    # ... существующие переменные ...
    PREFECT_API_URL: http://prefect-server:4200/api
```

**Новые файлы:**
- `pipelines/train.py` (Prefect flow)
- `pipelines/tasks/data_tasks.py`
- `pipelines/tasks/training_tasks.py`
- `pipelines/tasks/mlflow_tasks.py`
- `pipelines/cli.py`

### 2.2 Dependencies

**Добавить в pyproject.toml:**
```toml
dependencies = [
    # ... существующие ...
    "prefect>=3.1.0",
]
```

**Установка:**
```bash
docker compose exec training uv add prefect>=3.1.0
# Или пересобрать контейнер после изменения pyproject.toml
docker compose up -d --build training
```

### 2.3 Training Flow Structure

**Файл:** `pipelines/train.py`

```python
from prefect import flow, task

@flow(name="sentiment-training", log_prints=True)
def training_flow(
    data_path: str = "data/imdb_sample.csv",
    experiment_name: str = "sentiment-classifier",
    register_model: bool = True,
    set_champion: bool = False,
):
    """
    Training pipeline with MLflow logging and Model Registry.

    Args:
        data_path: Path to training data
        experiment_name: MLflow experiment name
        register_model: Whether to register model in MLflow Registry
        set_champion: Whether to set 'champion' alias on new model
    """
    # Task 1: Load data
    data = load_data_task(data_path)

    # Task 2: Split data
    X_train, X_test, y_train, y_test = split_data_task(data)

    # Task 3: Create and train pipeline
    pipeline = train_model_task(X_train, y_train)

    # Task 4: Evaluate
    metrics = evaluate_task(pipeline, X_test, y_test)

    # Task 5: Log to MLflow
    run_id = log_to_mlflow_task(pipeline, metrics, experiment_name)

    # Task 6: Register model
    if register_model:
        version = register_model_task(run_id, model_name="sentiment-classifier")

        # Task 7: Set champion alias
        if set_champion:
            set_champion_alias_task(
                model_name="sentiment-classifier",
                version=version,
                metric="f1"
            )

    return {"run_id": run_id, "metrics": metrics}
```

### 2.4 Prefect Tasks

**`pipelines/tasks/data_tasks.py`:**
```python
import pandas as pd
from prefect import task

from src.data import load_dataset, split_data

@task(name="load-data", retries=2, retry_delay_seconds=10)
def load_data_task(data_path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    return load_dataset(data_path)

@task(name="split-data")
def split_data_task(data: pd.DataFrame, test_size: float = 0.2):
    """Split data into train/test sets."""
    train_df, test_df = split_data(data, test_size=test_size)
    return (
        train_df["text"], test_df["text"],
        train_df["label"], test_df["label"]
    )
```

**`pipelines/tasks/training_tasks.py`:**
```python
from prefect import task

from src.model import create_training_pipeline, train_model, evaluate_model

@task(name="train-model", log_prints=True)
def train_model_task(X_train, y_train):
    """Create and train sklearn pipeline."""
    pipeline = create_training_pipeline()
    return train_model(pipeline, X_train, y_train)

@task(name="evaluate-model")
def evaluate_task(pipeline, X_test, y_test) -> dict:
    """Evaluate model and return metrics."""
    return evaluate_model(pipeline, X_test, y_test)
```

**`pipelines/tasks/mlflow_tasks.py`:**
```python
import mlflow
from mlflow import MlflowClient
from prefect import task, get_run_logger

@task(name="log-mlflow")
def log_to_mlflow_task(pipeline, metrics: dict, experiment_name: str) -> str:
    """Log model and metrics to MLflow."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline, "model",
            input_example=["Great movie!", "Terrible film."]
        )
        return run.info.run_id

@task(name="register-model")
def register_model_task(run_id: str, model_name: str) -> int:
    """Register model in MLflow Model Registry."""
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"

    mv = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id
    )
    return int(mv.version)

@task(name="set-champion-alias")
def set_champion_alias_task(
    model_name: str,
    version: int,
    metric: str = "f1"
) -> bool:
    """
    Set 'champion' alias on model version.
    Compares with current champion and only updates if new model is better.
    """
    logger = get_run_logger()
    client = MlflowClient()

    # Get new model metrics
    new_version = client.get_model_version(model_name, str(version))
    new_run = client.get_run(new_version.run_id)
    new_metric = new_run.data.metrics.get(metric, 0)

    # Check if champion exists
    try:
        current = client.get_model_version_by_alias(model_name, "champion")
        current_run = client.get_run(current.run_id)
        current_metric = current_run.data.metrics.get(metric, 0)

        if new_metric <= current_metric:
            logger.info(
                f"New model ({new_metric:.4f}) not better than "
                f"champion ({current_metric:.4f}). Skipping."
            )
            return False
    except mlflow.exceptions.MlflowException:
        logger.info("No existing champion. Setting new one.")

    # Set champion alias
    client.set_registered_model_alias(model_name, "champion", version)
    logger.info(f"Set 'champion' alias on version {version}")
    return True
```

### 2.5 CLI Interface

**Файл:** `pipelines/cli.py`

```python
#!/usr/bin/env python3
"""CLI for training pipeline."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def train_command(args):
    """Run training flow."""
    from pipelines.train import training_flow

    result = training_flow(
        data_path=args.data_path,
        register_model=args.register,
        set_champion=args.champion,
    )
    print(f"Training completed. Run ID: {result['run_id']}")
    print(f"Metrics: {result['metrics']}")

def model_info_command(args):
    """Show current champion model info."""
    from mlflow import MlflowClient

    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(
            args.model_name, "champion"
        )
        run = client.get_run(version.run_id)

        print(f"Model: {args.model_name}")
        print(f"Version: {version.version}")
        print(f"Alias: champion")
        print(f"Run ID: {version.run_id}")
        print(f"Metrics:")
        for k, v in run.data.metrics.items():
            print(f"  {k}: {v:.4f}")
    except Exception as e:
        print(f"No champion model found: {e}")

def set_alias_command(args):
    """Manually set alias on model version."""
    from mlflow import MlflowClient

    client = MlflowClient()
    client.set_registered_model_alias(
        args.model_name, args.alias, args.version
    )
    print(f"Set alias '{args.alias}' on {args.model_name} v{args.version}")

def main():
    parser = argparse.ArgumentParser(description="Sentiment MLOps CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train command
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument(
        "--data-path", default="data/imdb_sample.csv"
    )
    train_parser.add_argument(
        "--register", action="store_true", default=True,
        help="Register model in MLflow Registry"
    )
    train_parser.add_argument(
        "--champion", action="store_true",
        help="Set as champion if better than current"
    )
    train_parser.set_defaults(func=train_command)

    # model-info command
    info_parser = subparsers.add_parser("model-info", help="Show champion model")
    info_parser.add_argument(
        "--model-name", default="sentiment-classifier"
    )
    info_parser.set_defaults(func=model_info_command)

    # set-alias command
    alias_parser = subparsers.add_parser("set-alias", help="Set model alias")
    alias_parser.add_argument("--model-name", required=True)
    alias_parser.add_argument("--version", type=int, required=True)
    alias_parser.add_argument("--alias", required=True)
    alias_parser.set_defaults(func=set_alias_command)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
```

### 2.6 Model Registry (Using Aliases)

> **ВАЖНО**: Model Registry Stages (Staging, Production, Archived) **deprecated** в MLflow 2.9+.
> Используем **Model Version Aliases**.

**Стратегия:**
- `@champion` — production модель
- `@challenger` — кандидат для A/B тестирования (опционально)

**Загрузка модели по alias:**
```python
model_uri = "models:/sentiment-classifier@champion"
model = mlflow.sklearn.load_model(model_uri)
```

### 2.7 Verification Checklist

**Команды для тестирования:**
```bash
# Запустить все сервисы
docker compose up -d --build

# Проверить Prefect UI
open http://localhost:4200

# Запустить training flow
docker compose exec training uv run python -m pipelines.cli train

# Запустить с регистрацией champion
docker compose exec training uv run python -m pipelines.cli train --champion

# Проверить champion модель
docker compose exec training uv run python -m pipelines.cli model-info

# Проверить в MLflow UI
open http://localhost:5001
```

**Checklist:**
- [ ] Prefect UI доступен на http://localhost:4200
- [ ] `train` command запускает flow
- [ ] Flow виден в Prefect UI с состоянием Completed
- [ ] Все tasks завершены успешно
- [ ] Модель зарегистрирована в MLflow Model Registry
- [ ] При `--champion` модель получает alias "champion"
- [ ] `model-info` показывает текущую champion модель
- [ ] Повторный запуск с `--champion` корректно сравнивает метрики

---

## Iteration 3: FastAPI Model Serving

**Цель**: Создать FastAPI сервис для inference с автоматической загрузкой champion модели и hot-reload.

### 3.1 FastAPI Application Structure

```
app/
├── __init__.py
├── main.py              # FastAPI app, lifespan, routes
├── config.py            # App-specific settings
├── schemas.py           # Pydantic models
├── model_loader.py      # Model loading from MLflow
├── dependencies.py      # FastAPI dependencies
└── exceptions.py        # Custom exceptions
```

### 3.2 Docker Configuration

**Файл:** `docker/fastapi/Dockerfile`

```dockerfile
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

# Copy and install dependencies
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --no-install-project 2>/dev/null || uv sync --no-dev --no-install-project

# Copy source
COPY src/ ./src/
COPY app/ ./app/

ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Добавить в docker-compose.yml:**
```yaml
fastapi:
  build:
    context: .
    dockerfile: docker/fastapi/Dockerfile
  container_name: sentiment-fastapi
  ports:
    - "8000:8000"
  environment:
    MLFLOW_TRACKING_URI: http://mlflow:5000
    MLFLOW_S3_ENDPOINT_URL: http://minio:9000
    AWS_ACCESS_KEY_ID: ${MINIO_ROOT_USER:-minioadmin}
    AWS_SECRET_ACCESS_KEY: ${MINIO_ROOT_PASSWORD:-minioadmin}
    MODEL_NAME: sentiment-classifier
    MODEL_RELOAD_INTERVAL: "60"
  depends_on:
    mlflow:
      condition: service_healthy
  networks:
    - mlops-network
  restart: unless-stopped
```

### 3.3 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Get sentiment prediction |
| `/model-info` | GET | Current model info |
| `/admin/reload` | POST | Force model reload |

**Schemas (`app/schemas.py`):**
```python
from pydantic import BaseModel
from typing import Literal
from datetime import datetime

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sentiment: Literal["positive", "negative"]
    confidence: float
    model_version: str

class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    model_loaded: bool

class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    alias: str
    run_id: str
    metrics: dict
```

### 3.4 Model Loader

**Файл:** `app/model_loader.py`

```python
import asyncio
import mlflow
from mlflow import MlflowClient
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    name: str
    version: str
    alias: str
    run_id: str
    metrics: dict

class ModelManager:
    def __init__(self, model_name: str = "sentiment-classifier"):
        self.model_name = model_name
        self.model = None
        self.info: ModelInfo | None = None
        self._lock = asyncio.Lock()

    def is_ready(self) -> bool:
        return self.model is not None

    async def load_champion(self) -> bool:
        """Load champion model. Returns True if model was updated."""
        async with self._lock:
            client = MlflowClient()

            try:
                version = client.get_model_version_by_alias(
                    self.model_name, "champion"
                )
            except Exception as e:
                logger.error(f"No champion model: {e}")
                return False

            # Skip if same version
            if self.info and self.info.version == version.version:
                return False

            # Load model
            model_uri = f"models:/{self.model_name}@champion"
            self.model = mlflow.sklearn.load_model(model_uri)

            # Get metrics
            run = client.get_run(version.run_id)

            self.info = ModelInfo(
                name=self.model_name,
                version=version.version,
                alias="champion",
                run_id=version.run_id,
                metrics=run.data.metrics
            )

            logger.info(f"Loaded model v{version.version}")
            return True

    def predict(self, text: str) -> tuple[str, float]:
        """Return (sentiment, confidence)."""
        proba = self.model.predict_proba([text])[0]
        pred = self.model.predict([text])[0]

        sentiment = "positive" if pred == 1 else "negative"
        confidence = float(max(proba))

        return sentiment, confidence
```

### 3.5 Main Application

**Файл:** `app/main.py`

```python
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import logging

from app.schemas import (
    PredictRequest, PredictResponse,
    HealthResponse, ModelInfoResponse
)
from app.model_loader import ModelManager
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_manager = ModelManager(settings.model_name)

async def reload_task():
    """Background task to check for new models."""
    while True:
        try:
            updated = await model_manager.load_champion()
            if updated:
                logger.info(f"Model reloaded: v{model_manager.info.version}")
        except Exception as e:
            logger.error(f"Reload failed: {e}")

        await asyncio.sleep(settings.model_reload_interval)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await model_manager.load_champion()
    task = asyncio.create_task(reload_task())
    yield
    # Shutdown
    task.cancel()

app = FastAPI(title="Sentiment API", lifespan=lifespan)

@app.get("/health", response_model=HealthResponse)
async def health():
    return {
        "status": "healthy" if model_manager.is_ready() else "unhealthy",
        "model_loaded": model_manager.is_ready()
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if not model_manager.is_ready():
        raise HTTPException(503, "Model not loaded")

    sentiment, confidence = model_manager.predict(request.text)

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "model_version": model_manager.info.version
    }

@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    if not model_manager.info:
        raise HTTPException(503, "Model not loaded")

    return {
        "model_name": model_manager.info.name,
        "version": model_manager.info.version,
        "alias": model_manager.info.alias,
        "run_id": model_manager.info.run_id,
        "metrics": model_manager.info.metrics
    }

@app.post("/admin/reload")
async def reload():
    updated = await model_manager.load_champion()
    return {
        "reloaded": updated,
        "version": model_manager.info.version if model_manager.info else None
    }
```

### 3.6 Dependencies

**Добавить в pyproject.toml:**
```toml
dependencies = [
    # ... существующие ...
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
]
```

### 3.7 Verification Checklist

**Команды:**
```bash
# Запустить все сервисы
docker compose up -d --build

# Проверить health
curl http://localhost:8000/health

# Получить предсказание
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie is amazing!"}'

# Информация о модели
curl http://localhost:8000/model-info

# Принудительная перезагрузка
curl -X POST http://localhost:8000/admin/reload
```

**Checklist:**
- [ ] FastAPI запускается в docker compose
- [ ] `/health` возвращает `{"status": "healthy"}`
- [ ] `/predict` возвращает корректный sentiment
- [ ] `/model-info` показывает version и metrics
- [ ] После training с `--champion` модель автоматически обновляется
- [ ] Health check в Docker работает

---

## Iteration 4: DVC, CI/CD & Production Readiness

**Цель**: Добавить версионирование данных, CI/CD, тесты.

### 4.1 DVC Setup

**Команды (выполнять в контейнере или локально):**
```bash
# Инициализация (локально)
uv add dvc[s3]>=3.58.0
uv run dvc init

# Настройка remote (MinIO)
uv run dvc remote add -d minio s3://data
uv run dvc remote modify minio endpointurl http://localhost:9000
uv run dvc remote modify minio access_key_id minioadmin
uv run dvc remote modify minio secret_access_key minioadmin

# Трекинг данных
uv run dvc add data/imdb_sample.csv
git add data/imdb_sample.csv.dvc data/.gitignore
```

### 4.2 GitHub Actions

**`.github/workflows/ci.yml`:**
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Lint
        run: |
          uv sync --dev
          uv run ruff check .
          uv run ruff format --check .

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Test
        run: |
          uv sync --dev
          uv run pytest tests/ -v

  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4

      - name: Build images
        run: docker compose build
```

**`.github/workflows/train.yml`:**
```yaml
name: Train Model

on:
  workflow_dispatch:
    inputs:
      set_champion:
        description: 'Set as champion'
        type: boolean
        default: false

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Start services
        run: docker compose up -d postgres minio mlflow training

      - name: Wait for healthy
        run: sleep 30 && docker compose ps

      - name: Train
        run: |
          docker compose exec -T training uv run python -m pipelines.cli train \
            ${{ inputs.set_champion && '--champion' || '' }}

      - name: Cleanup
        if: always()
        run: docker compose down -v
```

### 4.3 Tests

**Структура:**
```
tests/
├── __init__.py
├── conftest.py
├── unit/
│   ├── test_preprocessing.py
│   └── test_data.py
└── integration/
    └── test_api.py
```

**`tests/unit/test_preprocessing.py`:**
```python
from src.preprocessing import clean_text

def test_clean_text_lowercase():
    assert clean_text("HELLO World") == "hello world"

def test_clean_text_removes_html():
    assert clean_text("<br/>Hello<br/>") == "hello"

def test_clean_text_removes_punctuation():
    assert clean_text("Hello, World!") == "hello world"
```

### 4.4 Verification Checklist

- [ ] `uv run dvc pull` загружает данные
- [ ] `uv run dvc push` сохраняет данные
- [ ] CI проходит на push
- [ ] `uv run ruff check .` без ошибок
- [ ] Тесты проходят
- [ ] Manual workflow работает

---

## Summary

| Aspect | Iteration 1 ✅ | Iteration 2 | Iteration 3 | Iteration 4 |
|--------|---------------|-------------|-------------|-------------|
| **Focus** | Infrastructure | Pipeline | Serving | DevOps |
| **Services** | postgres, minio, mlflow, training | +prefect | +fastapi | - |
| **Key** | Basic training | Model Registry | Inference API | CI/CD |

---

## Technical Notes

### Ports

| Service | External | Internal |
|---------|----------|----------|
| MLflow | 5001 | 5000 |
| MinIO API | 9000 | 9000 |
| MinIO Console | 9001 | 9001 |
| Prefect | 4200 | 4200 |
| FastAPI | 8000 | 8000 |
| PostgreSQL | 5432 | 5432 |

### MLflow Aliases

```python
# Загрузка champion модели
model_uri = "models:/sentiment-classifier@champion"
model = mlflow.sklearn.load_model(model_uri)

# Назначение alias
client.set_registered_model_alias("sentiment-classifier", "champion", version)

# Получение по alias
version = client.get_model_version_by_alias("sentiment-classifier", "champion")
```

### Команды

```bash
# Все команды выполняются через docker compose exec
docker compose exec training uv run python pipelines/train_simple.py
docker compose exec training uv run python -m pipelines.cli train --champion
docker compose exec training uv run python pipelines/test_model.py
```
