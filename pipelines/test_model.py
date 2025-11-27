#!/usr/bin/env python3
"""
Test model predictions from MLflow.

Usage:
    python pipelines/test_model.py
    python pipelines/test_model.py --run-id <run_id>
    python pipelines/test_model.py --text "This movie is great!"
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import mlflow

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_latest_run_id() -> str:
    """Get the latest run ID from the experiment."""
    client = mlflow.MlflowClient()

    # Get experiment by name
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "sentiment-classifier")
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # Get latest run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1,
        order_by=["start_time DESC"],
    )

    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    return runs[0].info.run_id


def load_model(run_id: str):
    """Load model from MLflow."""
    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Loading model from: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


def predict(model, texts: list[str]) -> list[dict]:
    """Run predictions on texts."""
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)

    results = []
    for text, pred, proba in zip(texts, predictions, probabilities, strict=True):
        sentiment = "positive" if pred == 1 else "negative"
        confidence = float(max(proba))
        results.append({
            "text": text[:100] + "..." if len(text) > 100 else text,
            "sentiment": sentiment,
            "confidence": confidence,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Test sentiment model predictions")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLflow run ID (default: latest run)",
    )
    parser.add_argument(
        "--text",
        type=str,
        action="append",
        help="Text to predict (can be used multiple times)",
    )

    args = parser.parse_args()

    # Default test texts
    default_texts = [
        "This movie is absolutely fantastic! Best film I have ever seen.",
        "Terrible waste of time. Worst movie ever made.",
        "It was okay, nothing special but not bad either.",
        "The acting was superb and the story was captivating.",
        "I fell asleep halfway through. So boring!",
    ]

    texts = args.text if args.text else default_texts

    # Get run ID
    run_id = args.run_id or get_latest_run_id()
    logger.info(f"Using run ID: {run_id}")

    # Load model
    model = load_model(run_id)

    # Run predictions
    logger.info(f"Running predictions on {len(texts)} texts...")
    results = predict(model, texts)

    # Print results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)

    for r in results:
        emoji = "👍" if r["sentiment"] == "positive" else "👎"
        print(f"\n{emoji} {r['sentiment'].upper()} ({r['confidence']:.1%})")
        print(f"   \"{r['text']}\"")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
