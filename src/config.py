"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # PostgreSQL
    postgres_user: str = "mlflow"
    postgres_password: str = "mlflow"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "mlflow"

    # MinIO
    minio_root_user: str = "minioadmin"
    minio_root_password: str = "minioadmin"
    minio_endpoint: str = "http://localhost:9000"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_s3_endpoint_url: str = "http://localhost:9000"
    mlflow_experiment_name: str = "sentiment-classifier"

    # AWS credentials for MinIO
    aws_access_key_id: str = "minioadmin"
    aws_secret_access_key: str = "minioadmin"

    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = Path(__file__).parent.parent / "data"
    models_dir: Path = Path(__file__).parent.parent / "models"

    @property
    def database_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def configure_mlflow_environment(self) -> None:
        """Set environment variables required by MLflow for S3/MinIO access."""
        import os

        os.environ["MLFLOW_TRACKING_URI"] = self.mlflow_tracking_uri
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.mlflow_s3_endpoint_url
        os.environ["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
