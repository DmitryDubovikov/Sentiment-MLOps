# Minimal MLOps Project — Sentiment Classifier (scikit-learn + FastAPI + MLflow)

## 🎯 Goal
Build a minimal, production-like MLOps system that demonstrates the full lifecycle of a machine learning model using a simple sentiment-classification task.  
The project highlights essential MLOps skills for a Python backend engineer: experiment tracking, model versioning, automated pipelines, reproducibility, artifact storage, and automatic model deployment.

---

## 🚀 Project Summary

This project implements an end-to-end MLOps workflow for a sentiment analysis model (“positive / negative review classifier”) built with **scikit-learn**.

The system supports:

- **Experiment tracking & model registry** — MLflow  
- **Reproducible pipelines** — Prefect  
- **Versioned data storage** — DVC  
- **Model & artifact storage** — MinIO  
- **Model serving** — FastAPI  
- **Containerized environment** — Docker Compose  
- **CI/CD automation** — GitHub Actions  

Everything must run via `docker compose up`.

---

## 📦 Functional Requirements

### 1. Training Pipeline (Prefect)

A reproducible training pipeline must:

1. Load a small sentiment dataset (IMDb small or equivalent)
2. Preprocess text (cleaning + TF-IDF)
3. Train a `LogisticRegression` classifier
4. Compute metrics (accuracy, F1)
5. Log parameters, metrics, and artifacts to **MLflow**
6. Save the model to **MinIO** (S3 bucket)
7. Register the model in **MLflow Model Registry**
8. Optionally: auto-promote the best model to “Production”

Pipeline execution:
- manually via CLI  
- or via Prefect UI  
- optional trigger from GitHub Actions  

---

### 2. Model Serving (FastAPI)

A separate service must:

- Load the **current production model** from MinIO at startup
- Expose endpoints:

| Endpoint       | Description                       |
|----------------|-----------------------------------|
| `POST /predict` | Return sentiment prediction        |
| `GET /health`   | Health check                       |
| `GET /model-info` | Return model version & metadata from MLflow |

The service must support **automatic reload** when a new production model appears.

---

### 3. Infrastructure (Docker Compose)

The system includes:

- `fastapi-server` — inference API  
- `training-service` — Prefect pipeline  
- `mlflow` — MLflow tracking server  
- `minio` + MinIO console — artifact/model storage  
- `postgres` — MLflow backend store  
- `prefect-agent` (optional)

All components run locally using **Docker Compose**.

---

## 🗄 Data Management

- Raw dataset tracked via **DVC**
- DVC remote is stored in MinIO
- Preprocessed artifacts logged in MLflow

---

## 🔁 CI/CD (GitHub Actions)

GitHub Actions workflow must include:

- Linting (ruff)
- Optional unit tests
- Build and push Docker images
- Optional: trigger a Prefect training flow
- Auto-register new model versions in MLflow

---

## 🧰 Tech Stack

### Core
- Python 3.11+
- scikit-learn
- FastAPI
- Prefect
- MLflow

### Storage
- MinIO (S3)
- PostgreSQL (MLflow backend)

### Reproducibility
- DVC
- Docker & Docker Compose

### CI/CD
- GitHub Actions

---

## 📁 Repository Structure (expected)

```
.
├── app/
│   ├── main.py
│   ├── model_loader.py
│   ├── schemas.py
│   └── utils.py
│
├── pipelines/
│   ├── train.py
│   └── tasks/
│
├── data/                        # DVC dataset
├── models/                      # Local model cache
├── docker/
│
├── docker-compose.yml
├── requirements.txt
├── task_description.md
└── README.md (optional)
```

---

## ✔️ What This Project Demonstrates

- MLflow for experiments & model registry  
- DVC for versioned data  
- MinIO as object storage  
- Automated training via Prefect  
- Automated deployment of the best model  
- Reproducible environment through Docker  
- FastAPI for inference  
- GitHub Actions CI/CD workflow  

---

## 🏁 Outcome

A fully functional MLOps system that trains a model, tracks experiments, stores artifacts, deploys the latest production model, and exposes it via a clean FastAPI interface — all containerized and reproducible.
