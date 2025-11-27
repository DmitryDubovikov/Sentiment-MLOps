"""Data loading utilities for sentiment classification."""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    """
    Load sentiment dataset from CSV file.

    Expected format:
        - Column 'text': review text
        - Column 'label': 0 (negative) or 1 (positive)

    Args:
        data_path: Path to CSV file

    Returns:
        DataFrame with 'text' and 'label' columns
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    # Validate columns
    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns: {required_columns}. Found: {df.columns.tolist()}"
        )

    # Validate labels
    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Labels must be 0 or 1. Found: {unique_labels}")

    logger.info(f"Loaded dataset: {len(df)} samples from {data_path}")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test sets.

    Args:
        df: Input DataFrame with 'text' and 'label' columns
        test_size: Fraction of data for test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test")

    return train_df, test_df


def download_imdb_subset(
    output_path: str | Path,
    n_samples: int = 1000,
    random_state: int = 42,
) -> Path:
    """
    Download a subset of IMDb dataset from Hugging Face.

    Args:
        output_path: Path to save the CSV file
        n_samples: Number of samples to download (will be split 50/50 between classes)
        random_state: Random seed for reproducibility

    Returns:
        Path to the saved CSV file
    """
    from datasets import load_dataset as hf_load_dataset

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"Dataset already exists: {output_path}")
        return output_path

    logger.info("Downloading IMDb dataset from Hugging Face...")

    # Load IMDb dataset
    dataset = hf_load_dataset("imdb", split="train")

    # Convert to DataFrame
    df = pd.DataFrame({"text": dataset["text"], "label": dataset["label"]})

    # Sample balanced subset
    samples_per_class = n_samples // 2

    positive = df[df["label"] == 1].sample(n=samples_per_class, random_state=random_state)
    negative = df[df["label"] == 0].sample(n=samples_per_class, random_state=random_state)

    subset = pd.concat([positive, negative], ignore_index=True)
    subset = subset.sample(frac=1, random_state=random_state).reset_index(drop=True)  # Shuffle

    # Save to CSV
    subset.to_csv(output_path, index=False)
    logger.info(f"Saved {len(subset)} samples to {output_path}")

    return output_path
