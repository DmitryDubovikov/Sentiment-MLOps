#!/usr/bin/env python3
"""
Simple training script for sentiment classifier.

This script validates the infrastructure by:
1. Loading the IMDb dataset
2. Preprocessing text with TF-IDF
3. Training LogisticRegression
4. Evaluating metrics
5. Logging everything to MLflow

Usage:
    python pipelines/train_simple.py
    python pipelines/train_simple.py --data-path data/imdb_sample.csv
    python pipelines/train_simple.py --download  # Download dataset first
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_settings  # noqa: E402
from src.data import download_imdb_subset, load_dataset, split_data  # noqa: E402
from src.model import (  # noqa: E402
    create_training_pipeline,
    evaluate_model,
    get_pipeline_params,
    log_model_to_mlflow,
    train_model,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(
    data_path: str | None = None,
    download: bool = False,
    n_samples: int = 1000,
) -> str:
    """
    Run the training pipeline.

    Args:
        data_path: Path to dataset CSV
        download: Whether to download dataset first
        n_samples: Number of samples to download

    Returns:
        MLflow run ID
    """
    settings = get_settings()

    # Configure MLflow environment variables
    settings.configure_mlflow_environment()

    # Determine data path
    data_path = settings.data_dir / "imdb_sample.csv" if data_path is None else Path(data_path)

    # Download dataset if needed
    if download or not data_path.exists():
        logger.info("Downloading IMDb dataset...")
        download_imdb_subset(data_path, n_samples=n_samples)

    # Load and split data
    logger.info(f"Loading dataset from {data_path}")
    df = load_dataset(data_path)
    train_df, test_df = split_data(df, test_size=0.2)

    # Extract features and labels
    X_train, y_train = train_df["text"], train_df["label"]
    X_test, y_test = test_df["text"], test_df["label"]

    # Create and train pipeline
    logger.info("Creating training pipeline...")
    pipeline = create_training_pipeline(
        vectorizer_params={
            "max_features": 5000,
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95,
        },
        model_params={
            "C": 1.0,
            "max_iter": 1000,
        },
    )

    logger.info("Training model...")
    pipeline = train_model(pipeline, X_train, y_train)

    # Evaluate
    logger.info("Evaluating model...")
    metrics = evaluate_model(pipeline, X_test, y_test)

    # Get parameters for logging
    params = get_pipeline_params(pipeline)
    params["train_samples"] = len(X_train)
    params["test_samples"] = len(X_test)
    params["data_path"] = str(data_path)

    # Log to MLflow
    logger.info("Logging to MLflow...")
    run_id = log_model_to_mlflow(
        pipeline=pipeline,
        metrics=metrics,
        params=params,
        experiment_name=settings.mlflow_experiment_name,
        input_example=["This movie is amazing!", "Terrible waste of time."],
    )

    logger.info("Training completed successfully!")
    logger.info(f"MLflow run ID: {run_id}")
    logger.info(f"Metrics: {metrics}")
    logger.info(f"View results at: {settings.mlflow_tracking_uri}")

    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment classifier")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to dataset CSV (default: data/imdb_sample.csv)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download IMDb dataset before training",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples to download (default: 1000)",
    )

    args = parser.parse_args()

    try:
        run_id = main(
            data_path=args.data_path,
            download=args.download,
            n_samples=args.n_samples,
        )
        print(f"\n✓ Training completed. Run ID: {run_id}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
