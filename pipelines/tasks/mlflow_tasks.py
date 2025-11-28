"""Prefect tasks for MLflow logging and Model Registry."""

import logging

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from prefect import get_run_logger, task
from sklearn.pipeline import Pipeline

from src.model import get_pipeline_params

logger = logging.getLogger(__name__)


@task(name="log-mlflow")
def log_to_mlflow_task(
    pipeline: Pipeline,
    metrics: dict[str, float],
    experiment_name: str,
    data_path: str,
    train_samples: int,
    test_samples: int,
) -> str:
    """
    Log model and metrics to MLflow.

    Args:
        pipeline: Trained sklearn Pipeline
        metrics: Evaluation metrics
        experiment_name: Name of MLflow experiment
        data_path: Path to training data
        train_samples: Number of training samples
        test_samples: Number of test samples

    Returns:
        MLflow run ID
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        # Get and log parameters
        params = get_pipeline_params(pipeline)
        params["train_samples"] = train_samples
        params["test_samples"] = test_samples
        params["data_path"] = data_path
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Prepare signature and input example
        input_example = ["This is a great movie!", "Terrible film, waste of time."]
        signature = infer_signature(
            model_input=input_example,
            model_output=pipeline.predict(input_example),
        )

        # Log model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",  # Changed from artifact_path (deprecated)
            signature=signature,
            input_example=input_example,
            pip_requirements=["scikit-learn", "pandas", "numpy"],  # Explicit pip deps
        )

        return run.info.run_id


@task(name="register-model")
def register_model_task(run_id: str, model_name: str) -> int:
    """
    Register model in MLflow Model Registry.

    Args:
        run_id: MLflow run ID
        model_name: Name for registered model

    Returns:
        Model version number
    """
    prefect_logger = get_run_logger()
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"

    # Ensure registered model exists
    try:
        client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(model_name)
        prefect_logger.info(f"Created registered model: {model_name}")

    # Create model version
    mv = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
    )

    prefect_logger.info(f"Registered model version {mv.version}")
    return int(mv.version)


@task(name="set-champion-alias")
def set_champion_alias_task(
    model_name: str,
    version: int,
    metric: str = "f1",
) -> bool:
    """
    Set 'champion' alias on model version.

    Compares with current champion and only updates if new model is better.

    Args:
        model_name: Name of registered model
        version: Model version number
        metric: Metric to compare (default: f1)

    Returns:
        True if alias was set, False otherwise
    """
    prefect_logger = get_run_logger()
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
            prefect_logger.info(
                f"New model ({new_metric:.4f}) not better than "
                f"champion ({current_metric:.4f}). Skipping alias update."
            )
            return False
        else:
            prefect_logger.info(
                f"New model ({new_metric:.4f}) better than "
                f"champion ({current_metric:.4f}). Updating alias."
            )
    except mlflow.exceptions.MlflowException:
        prefect_logger.info("No existing champion. Setting new one.")

    # Set champion alias
    client.set_registered_model_alias(model_name, "champion", version)
    prefect_logger.info(f"Set 'champion' alias on version {version}")
    return True
