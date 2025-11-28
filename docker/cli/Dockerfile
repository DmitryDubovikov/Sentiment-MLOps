# Universal CLI container for all development tasks
# Usage: docker compose run --rm cli <command>
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy project files
COPY pyproject.toml uv.lock* ./

# Install ALL dependencies (including dev for testing/linting)
RUN uv sync --frozen 2>/dev/null || uv sync

# Copy source code
COPY src/ ./src/
COPY app/ ./app/
COPY pipelines/ ./pipelines/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Copy DVC config
COPY dvc.yaml params.yaml ./
COPY .dvc/ ./.dvc/

# Create data directory
RUN mkdir -p data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command - show help
CMD ["uv", "run", "python", "-m", "pipelines.cli", "--help"]
