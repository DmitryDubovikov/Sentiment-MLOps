"""Prefect tasks for model training and evaluation."""

import numpy as np
import pandas as pd
from prefect import task
from sklearn.pipeline import Pipeline

from src.model import create_training_pipeline, evaluate_model, train_model


@task(name="train-model", log_prints=True)
def train_model_task(
    X_train: pd.Series | list[str],
    y_train: pd.Series | np.ndarray,
    vectorizer_params: dict | None = None,
    model_params: dict | None = None,
) -> Pipeline:
    """
    Create and train sklearn pipeline.

    Args:
        X_train: Training texts
        y_train: Training labels
        vectorizer_params: Parameters for TF-IDF vectorizer
        model_params: Parameters for LogisticRegression

    Returns:
        Fitted sklearn Pipeline
    """
    pipeline = create_training_pipeline(
        vectorizer_params=vectorizer_params,
        model_params=model_params,
    )
    return train_model(pipeline, X_train, y_train)


@task(name="evaluate-model")
def evaluate_task(
    pipeline: Pipeline,
    X_test: pd.Series | list[str],
    y_test: pd.Series | np.ndarray,
) -> dict[str, float]:
    """
    Evaluate model and return metrics.

    Args:
        pipeline: Fitted sklearn Pipeline
        X_test: Test texts
        y_test: Test labels

    Returns:
        Dictionary of metrics (accuracy, f1, precision, recall)
    """
    return evaluate_model(pipeline, X_test, y_test)
