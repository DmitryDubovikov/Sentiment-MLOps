"""Model training and evaluation utilities."""

import logging
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from src.preprocessing import create_vectorizer, get_vectorizer_params

logger = logging.getLogger(__name__)


def create_model(
    C: float = 1.0,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    random_state: int = 42,
) -> LogisticRegression:
    """
    Create LogisticRegression classifier.

    Args:
        C: Inverse of regularization strength
        max_iter: Maximum iterations for convergence
        solver: Algorithm for optimization
        random_state: Random seed

    Returns:
        Configured LogisticRegression model
    """
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        random_state=random_state,
        n_jobs=-1,
    )

    logger.info(f"Created LogisticRegression: C={C}, max_iter={max_iter}, solver={solver}")

    return model


def create_training_pipeline(
    vectorizer_params: dict | None = None,
    model_params: dict | None = None,
) -> Pipeline:
    """
    Create full training pipeline with vectorizer and classifier.

    Args:
        vectorizer_params: Parameters for TF-IDF vectorizer
        model_params: Parameters for LogisticRegression

    Returns:
        sklearn Pipeline with vectorizer and classifier
    """
    vectorizer_params = vectorizer_params or {}
    model_params = model_params or {}

    vectorizer = create_vectorizer(**vectorizer_params)
    model = create_model(**model_params)

    pipeline = Pipeline(
        [
            ("tfidf", vectorizer),
            ("classifier", model),
        ]
    )

    return pipeline


def train_model(
    pipeline: Pipeline,
    X_train: pd.Series | list[str],
    y_train: pd.Series | np.ndarray,
) -> Pipeline:
    """
    Train the pipeline on training data.

    Args:
        pipeline: sklearn Pipeline with vectorizer and classifier
        X_train: Training texts
        y_train: Training labels

    Returns:
        Fitted pipeline
    """
    logger.info(f"Training model on {len(X_train)} samples...")

    pipeline.fit(X_train, y_train)

    logger.info("Model training completed")

    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.Series | list[str],
    y_test: pd.Series | np.ndarray,
) -> dict[str, float]:
    """
    Evaluate model performance on test data.

    Args:
        pipeline: Fitted sklearn Pipeline
        X_test: Test texts
        y_test: Test labels

    Returns:
        Dictionary of metrics
    """
    logger.info(f"Evaluating model on {len(X_test)} samples...")

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="binary"),
        "precision": precision_score(y_test, y_pred, average="binary"),
        "recall": recall_score(y_test, y_pred, average="binary"),
    }

    logger.info(f"Evaluation metrics: {metrics}")

    return metrics


def get_pipeline_params(pipeline: Pipeline) -> dict[str, Any]:
    """
    Extract all parameters from pipeline for logging.

    Args:
        pipeline: sklearn Pipeline

    Returns:
        Dictionary of all parameters
    """
    params = {}

    # Vectorizer params
    vectorizer = pipeline.named_steps.get("tfidf")
    if vectorizer:
        params.update(get_vectorizer_params(vectorizer))

    # Model params
    classifier = pipeline.named_steps.get("classifier")
    if classifier:
        params.update(
            {
                "model_C": classifier.C,
                "model_max_iter": classifier.max_iter,
                "model_solver": classifier.solver,
            }
        )

    return params


def log_model_to_mlflow(
    pipeline: Pipeline,
    metrics: dict[str, float],
    params: dict[str, Any],
    experiment_name: str,
    input_example: list[str] | None = None,
) -> str:
    """
    Log model, metrics, and parameters to MLflow.

    Args:
        pipeline: Trained sklearn Pipeline
        metrics: Evaluation metrics
        params: Model and vectorizer parameters
        experiment_name: Name of MLflow experiment
        input_example: Example input for model signature

    Returns:
        MLflow run ID
    """
    # Set experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Prepare signature and input example
        if input_example is None:
            input_example = ["This is a great movie!", "Terrible film, waste of time."]

        signature = infer_signature(
            model_input=input_example,
            model_output=pipeline.predict(input_example),
        )

        # Log model with signature
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=None,  # Don't register yet (will do in iteration 2)
        )

        run_id = run.info.run_id
        logger.info(f"Logged model to MLflow: run_id={run_id}")

        return run_id
