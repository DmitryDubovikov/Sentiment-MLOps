# Dockerfile for running tests in container
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY app/ ./app/
COPY pipelines/ ./pipelines/
COPY tests/ ./tests/

# Install dependencies including dev
RUN uv sync --dev

# Run tests by default
CMD ["uv", "run", "pytest", "tests/", "-v", "--cov=src", "--cov=app", "--cov=pipelines", "--cov-report=term-missing"]
