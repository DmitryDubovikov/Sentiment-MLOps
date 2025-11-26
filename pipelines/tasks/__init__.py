"""Prefect tasks for sentiment classification pipeline."""

from pipelines.tasks.data_tasks import load_data_task, split_data_task
from pipelines.tasks.mlflow_tasks import (
    log_to_mlflow_task,
    register_model_task,
    set_champion_alias_task,
)
from pipelines.tasks.training_tasks import evaluate_task, train_model_task

__all__ = [
    "load_data_task",
    "split_data_task",
    "train_model_task",
    "evaluate_task",
    "log_to_mlflow_task",
    "register_model_task",
    "set_champion_alias_task",
]
