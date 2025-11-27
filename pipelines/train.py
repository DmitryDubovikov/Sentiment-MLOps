"""
Prefect training flow for sentiment classifier.

This flow orchestrates the training pipeline with:
1. Data loading and splitting
2. Model training and evaluation
3. MLflow logging and model registration
4. Champion model promotion

Usage:
    # Via CLI
    python -m pipelines.cli train
    python -m pipelines.cli train --champion

    # Direct import
    from pipelines.train import training_flow
    result = training_flow(data_path="data/imdb_sample.csv", set_champion=True)
"""

import logging
import sys
from pathlib import Path

from prefect import flow

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipelines.tasks.data_tasks import load_data_task, split_data_task  # noqa: E402
from pipelines.tasks.mlflow_tasks import (  # noqa: E402
    log_to_mlflow_task,
    register_model_task,
    set_champion_alias_task,
)
from pipelines.tasks.training_tasks import evaluate_task, train_model_task  # noqa: E402
from src.config import get_settings  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@flow(name="sentiment-training", log_prints=True)
def training_flow(
    data_path: str = "data/imdb_sample.csv",
    experiment_name: str | None = None,
    model_name: str = "sentiment-classifier",
    register_model: bool = True,
    set_champion: bool = False,
    vectorizer_params: dict | None = None,
    model_params: dict | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
) -> dict:
    """
    Training pipeline with MLflow logging and Model Registry.

    Args:
        data_path: Path to training data CSV
        experiment_name: MLflow experiment name (default from settings)
        model_name: Name for registered model
        register_model: Whether to register model in MLflow Registry
        set_champion: Whether to set 'champion' alias on new model
        vectorizer_params: Parameters for TF-IDF vectorizer
        model_params: Parameters for LogisticRegression
        test_size: Fraction of data for test set (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        Dictionary with run_id, metrics, and optional version info
    """
    # Get settings
    settings = get_settings()
    settings.configure_mlflow_environment()

    if experiment_name is None:
        experiment_name = settings.mlflow_experiment_name

    # Default parameters
    if vectorizer_params is None:
        vectorizer_params = {
            "max_features": 5000,
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95,
        }
    if model_params is None:
        model_params = {
            "C": 1.0,
            "max_iter": 1000,
        }
    if test_size is None:
        test_size = 0.2
    if random_state is None:
        random_state = 42

    print(f"Starting training flow with data: {data_path}")

    # Task 1: Load data
    data = load_data_task(data_path)

    # Task 2: Split data
    X_train, X_test, y_train, y_test = split_data_task(
        data, test_size=test_size, random_state=random_state
    )

    # Task 3: Create and train pipeline
    pipeline = train_model_task(
        X_train,
        y_train,
        vectorizer_params=vectorizer_params,
        model_params=model_params,
    )

    # Task 4: Evaluate
    metrics = evaluate_task(pipeline, X_test, y_test)

    print(f"Metrics: {metrics}")

    # Task 5: Log to MLflow
    run_id = log_to_mlflow_task(
        pipeline=pipeline,
        metrics=metrics,
        experiment_name=experiment_name,
        data_path=data_path,
        train_samples=len(X_train),
        test_samples=len(X_test),
    )

    print(f"MLflow run ID: {run_id}")

    result = {
        "run_id": run_id,
        "metrics": metrics,
        "registered": False,
        "version": None,
        "is_champion": False,
    }

    # Task 6: Register model
    if register_model:
        version = register_model_task(run_id, model_name=model_name)
        result["registered"] = True
        result["version"] = version
        print(f"Registered model version: {version}")

        # Task 7: Set champion alias
        if set_champion:
            is_champion = set_champion_alias_task(
                model_name=model_name,
                version=version,
                metric="f1",
            )
            result["is_champion"] = is_champion

    return result


if __name__ == "__main__":
    # Run with default parameters
    result = training_flow()
    print(f"\nTraining completed: {result}")
