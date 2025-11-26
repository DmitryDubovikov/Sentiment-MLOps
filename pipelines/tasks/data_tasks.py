"""Prefect tasks for data loading and processing."""

import pandas as pd
from prefect import task

from src.data import load_dataset, split_data


@task(name="load-data", retries=2, retry_delay_seconds=10)
def load_data_task(data_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.

    Args:
        data_path: Path to the CSV file

    Returns:
        DataFrame with 'text' and 'label' columns
    """
    return load_dataset(data_path)


@task(name="split-data")
def split_data_task(
    data: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train/test sets.

    Args:
        data: DataFrame with 'text' and 'label' columns
        test_size: Fraction of data for test set

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    train_df, test_df = split_data(data, test_size=test_size)
    return (
        train_df["text"],
        test_df["text"],
        train_df["label"],
        test_df["label"],
    )
