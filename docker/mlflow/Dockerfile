FROM python:3.14-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    mlflow>=2.19.0 \
    psycopg2-binary>=2.9.0 \
    boto3>=1.35.0

# Expose MLflow port
EXPOSE 5000

# Start MLflow server (shell form to expand env variables)
# --allowed-hosts: allow requests from Docker network and localhost
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" \
    --serve-artifacts \
    --allowed-hosts "mlflow,mlflow:5000,localhost,localhost:5000,localhost:5001,127.0.0.1"
