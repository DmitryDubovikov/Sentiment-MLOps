"""Pydantic models for FastAPI request/response schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request schema for prediction endpoint."""

    text: str = Field(..., min_length=1, description="Text to analyze for sentiment")


class PredictResponse(BaseModel):
    """Response schema for prediction endpoint."""

    sentiment: Literal["positive", "negative"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: Literal["healthy", "unhealthy"]
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Response schema for model info endpoint."""

    model_name: str
    version: str
    alias: str
    run_id: str
    metrics: dict


class ReloadResponse(BaseModel):
    """Response schema for model reload endpoint."""

    reloaded: bool
    version: str | None
