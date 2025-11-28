"""FastAPI application for sentiment classification inference."""

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.exceptions import ChampionNotFoundError, ModelLoadError, ModelNotLoadedError
from app.model_loader import ModelManager
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    ReloadResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Reduce noisy logs from dependencies
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)  # retry warnings
logging.getLogger("mlflow").setLevel(logging.WARNING)  # progress bars, debug info

# Configure MLflow environment
settings.configure_mlflow_environment()

# Initialize model manager
model_manager = ModelManager(settings.model_name)


async def reload_task():
    """Background task to periodically check for new champion models."""
    while True:
        try:
            updated = await model_manager.load_champion()
            if updated:
                logger.info(f"Model hot-reloaded: v{model_manager.info.version}")
        except ChampionNotFoundError:
            logger.warning("No champion model available yet")
        except ModelLoadError as e:
            logger.error(f"Model reload failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during model reload: {e}")

        await asyncio.sleep(settings.model_reload_interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup: Load initial model
    logger.info("Starting FastAPI application...")
    try:
        await model_manager.load_champion()
        logger.info(f"Initial model loaded: v{model_manager.info.version}")
    except ChampionNotFoundError:
        logger.warning(
            "No champion model available at startup. API will return 503 until model is available."
        )
    except Exception as e:
        logger.error(f"Failed to load initial model: {e}")

    # Start background reload task
    task = asyncio.create_task(reload_task())
    logger.info(f"Background reload task started (interval: {settings.model_reload_interval}s)")

    yield

    # Shutdown: Cancel background task
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    logger.info("FastAPI application shutdown complete")


app = FastAPI(
    title="Sentiment Classification API",
    description="API for sentiment analysis using ML model served from MLflow Model Registry",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """
    Health check endpoint.

    Returns the health status of the API and whether a model is loaded.
    """
    return HealthResponse(
        status="healthy" if model_manager.is_ready() else "unhealthy",
        model_loaded=model_manager.is_ready(),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """
    Get sentiment prediction for input text.

    Returns:
        - sentiment: "positive" or "negative"
        - confidence: probability score (0.0 to 1.0)
        - model_version: version of the model used for prediction
    """
    if not model_manager.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for model to be available.",
        )

    try:
        sentiment, confidence = model_manager.predict(request.text)
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="Model not loaded") from None
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error") from e

    return PredictResponse(
        sentiment=sentiment,
        confidence=round(confidence, 4),
        model_version=model_manager.info.version,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """
    Get information about the currently loaded model.

    Returns model name, version, alias, run ID, and metrics.
    """
    if not model_manager.info:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. No model information available.",
        )

    return ModelInfoResponse(
        model_name=model_manager.info.name,
        version=model_manager.info.version,
        alias=model_manager.info.alias,
        run_id=model_manager.info.run_id,
        metrics=model_manager.info.metrics,
    )


@app.post("/admin/reload", response_model=ReloadResponse, tags=["Admin"])
async def reload():
    """
    Force reload of the champion model.

    Manually triggers a model reload from MLflow Model Registry.
    Returns whether a new model was loaded.
    """
    try:
        updated = await model_manager.load_champion()
        return ReloadResponse(
            reloaded=updated,
            version=model_manager.info.version if model_manager.info else None,
        )
    except ChampionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ModelLoadError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
