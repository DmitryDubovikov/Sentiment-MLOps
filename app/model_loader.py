"""Model loader for fetching champion model from MLflow Model Registry."""

import asyncio
import logging
from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient

from app.exceptions import ChampionNotFoundError, ModelLoadError

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Container for model metadata."""

    name: str
    version: str
    alias: str
    run_id: str
    metrics: dict


class ModelManager:
    """
    Manages model loading and hot-reloading from MLflow Model Registry.

    Uses the 'champion' alias to identify the production model.
    """

    def __init__(self, model_name: str = "sentiment-classifier"):
        self.model_name = model_name
        self.model = None
        self.info: ModelInfo | None = None
        self._lock = asyncio.Lock()

    def is_ready(self) -> bool:
        """Check if a model is loaded and ready for inference."""
        return self.model is not None

    async def load_champion(self) -> bool:
        """
        Load the champion model from MLflow Model Registry.

        Returns:
            True if a new model was loaded, False if already up-to-date.

        Raises:
            ChampionNotFoundError: If no champion model exists.
            ModelLoadError: If model loading fails.
        """
        async with self._lock:
            client = MlflowClient()

            # Get champion model version
            try:
                version = client.get_model_version_by_alias(self.model_name, "champion")
            except Exception as e:
                logger.error(f"Failed to get champion model: {e}")
                raise ChampionNotFoundError(
                    f"No champion model found for '{self.model_name}'"
                ) from e

            # Skip if same version is already loaded
            if self.info and self.info.version == version.version:
                logger.debug(f"Model v{version.version} already loaded")
                return False

            # Load model from MLflow
            try:
                model_uri = f"models:/{self.model_name}@champion"
                self.model = mlflow.sklearn.load_model(model_uri)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise ModelLoadError(f"Failed to load model from '{model_uri}'") from e

            # Get run metrics
            try:
                run = client.get_run(version.run_id)
                metrics = dict(run.data.metrics)
            except Exception as e:
                logger.warning(f"Failed to get run metrics: {e}")
                metrics = {}

            # Update model info
            self.info = ModelInfo(
                name=self.model_name,
                version=version.version,
                alias="champion",
                run_id=version.run_id,
                metrics=metrics,
            )

            logger.info(f"Loaded model v{version.version} (run_id: {version.run_id})")
            return True

    def predict(self, text: str) -> tuple[str, float]:
        """
        Get sentiment prediction for text.

        Args:
            text: Input text to classify.

        Returns:
            Tuple of (sentiment, confidence) where sentiment is 'positive' or 'negative'.

        Raises:
            ModelNotLoadedError: If no model is loaded.
        """
        if not self.is_ready():
            from app.exceptions import ModelNotLoadedError

            raise ModelNotLoadedError("No model loaded")

        # Get prediction and probability
        proba = self.model.predict_proba([text])[0]
        pred = self.model.predict([text])[0]

        sentiment = "positive" if pred == 1 else "negative"
        confidence = float(max(proba))

        return sentiment, confidence

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """
        Get sentiment predictions for multiple texts.

        Args:
            texts: List of input texts to classify.

        Returns:
            List of (sentiment, confidence) tuples.
        """
        if not self.is_ready():
            from app.exceptions import ModelNotLoadedError

            raise ModelNotLoadedError("No model loaded")

        probas = self.model.predict_proba(texts)
        preds = self.model.predict(texts)

        results = []
        for pred, proba in zip(preds, probas):
            sentiment = "positive" if pred == 1 else "negative"
            confidence = float(max(proba))
            results.append((sentiment, confidence))

        return results
