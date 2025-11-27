"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "Terrible film. Complete waste of time and money.",
        "The acting was great but the plot was confusing.",
        "One of the best movies I've ever seen!",
        "I couldn't even finish watching this garbage.",
        "Amazing cinematography and storytelling.",
        "Worst movie I have ever watched in my life.",
        "Highly recommended for all movie lovers.",
        "Do not waste your money on this trash.",
        "A masterpiece of modern cinema.",
    ]


@pytest.fixture
def sample_labels():
    """Sample labels for testing (0=negative, 1=positive)."""
    return [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]


@pytest.fixture
def sample_dataframe(sample_texts, sample_labels):
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        "text": sample_texts,
        "label": sample_labels,
    })


@pytest.fixture
def large_sample_dataframe():
    """Larger sample DataFrame for training tests."""
    positive_texts = [
        "This is amazing!",
        "Great product, highly recommend.",
        "Excellent quality and fast shipping.",
        "Love it! Will buy again.",
        "Perfect, exactly what I needed.",
        "Outstanding service and quality.",
        "Best purchase I've made this year.",
        "Fantastic experience overall.",
        "Wonderful product, exceeded expectations.",
        "Highly satisfied with this purchase.",
    ] * 5

    negative_texts = [
        "Terrible quality, very disappointed.",
        "Waste of money, don't buy.",
        "Awful experience, never again.",
        "Product broke after one day.",
        "Complete garbage, avoid at all costs.",
        "Worst purchase ever made.",
        "Absolutely horrible, total scam.",
        "Defective product, terrible service.",
        "Very poor quality, not recommended.",
        "Disappointed with this purchase.",
    ] * 5

    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)

    return pd.DataFrame({"text": texts, "label": labels})


@pytest.fixture
def temp_csv_file(sample_dataframe):
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_dataframe.to_csv(f, index=False)
        temp_path = Path(f.name)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing without actual MLflow connection."""
    mock = MagicMock()
    mock.set_experiment = MagicMock()
    mock.start_run = MagicMock()
    mock.log_params = MagicMock()
    mock.log_metrics = MagicMock()
    mock.sklearn.log_model = MagicMock()
    return mock


@pytest.fixture
def mock_model_manager():
    """Mock ModelManager for API testing."""
    manager = MagicMock()
    manager.is_ready.return_value = True
    manager.predict.return_value = ("positive", 0.95)
    manager.info = MagicMock()
    manager.info.name = "sentiment-classifier"
    manager.info.version = "1"
    manager.info.alias = "champion"
    manager.info.run_id = "test-run-id"
    manager.info.metrics = {"accuracy": 0.92, "f1": 0.91}
    return manager
