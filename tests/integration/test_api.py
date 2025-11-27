"""Integration tests for FastAPI endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.exceptions import ChampionNotFoundError


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_with_model_loaded(self, mock_model_manager):
        """Test health endpoint when model is loaded."""
        with patch("app.main.model_manager", mock_model_manager):
            with patch("app.main.reload_task", AsyncMock()):
                from app.main import app

                client = TestClient(app, raise_server_exceptions=False)
                response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_without_model(self, mock_model_manager):
        """Test health endpoint when model is not loaded."""
        mock_model_manager.is_ready.return_value = False

        with patch("app.main.model_manager", mock_model_manager):
            with patch("app.main.reload_task", AsyncMock()):
                from app.main import app

                client = TestClient(app, raise_server_exceptions=False)
                response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_positive(self, mock_model_manager):
        """Test prediction for positive sentiment."""
        mock_model_manager.predict.return_value = ("positive", 0.95)

        with patch("app.main.model_manager", mock_model_manager):
            with patch("app.main.reload_task", AsyncMock()):
                from app.main import app

                client = TestClient(app, raise_server_exceptions=False)
                response = client.post(
                    "/predict",
                    json={"text": "This movie is amazing!"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "positive"
        assert data["confidence"] == 0.95
        assert "model_version" in data

    def test_predict_negative(self, mock_model_manager):
        """Test prediction for negative sentiment."""
        mock_model_manager.predict.return_value = ("negative", 0.88)

        with patch("app.main.model_manager", mock_model_manager):
            with patch("app.main.reload_task", AsyncMock()):
                from app.main import app

                client = TestClient(app, raise_server_exceptions=False)
                response = client.post(
                    "/predict",
                    json={"text": "This movie is terrible!"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "negative"
        assert data["confidence"] == 0.88

    def test_predict_model_not_loaded(self, mock_model_manager):
        """Test prediction when model is not loaded."""
        mock_model_manager.is_ready.return_value = False

        with patch("app.main.model_manager", mock_model_manager):
            with patch("app.main.reload_task", AsyncMock()):
                from app.main import app

                client = TestClient(app, raise_server_exceptions=False)
                response = client.post(
                    "/predict",
                    json={"text": "Test text"},
                )

        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    def test_predict_invalid_request(self, mock_model_manager):
        """Test prediction with invalid request body."""
        with patch("app.main.model_manager", mock_model_manager):
            with patch("app.main.reload_task", AsyncMock()):
                from app.main import app

                client = TestClient(app, raise_server_exceptions=False)
                response = client.post("/predict", json={})

        assert response.status_code == 422  # Validation error

    def test_predict_empty_text(self, mock_model_manager):
        """Test prediction with empty text returns validation error."""
        with patch("app.main.model_manager", mock_model_manager):
            with patch("app.main.reload_task", AsyncMock()):
                from app.main import app

                client = TestClient(app, raise_server_exceptions=False)
                response = client.post("/predict", json={"text": ""})

        # Empty text should fail validation (min_length=1 in schema)
        assert response.status_code == 422


class TestModelInfoEndpoint:
    """Tests for /model-info endpoint."""

    def test_model_info_success(self, mock_model_manager):
        """Test model info endpoint with loaded model."""
        with patch("app.main.model_manager", mock_model_manager):
            with patch("app.main.reload_task", AsyncMock()):
                from app.main import app

                client = TestClient(app, raise_server_exceptions=False)
                response = client.get("/model-info")

        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "sentiment-classifier"
        assert data["version"] == "1"
        assert data["alias"] == "champion"
        assert data["run_id"] == "test-run-id"
        assert "metrics" in data

    def test_model_info_not_loaded(self, mock_model_manager):
        """Test model info when no model is loaded."""
        mock_model_manager.info = None

        with patch("app.main.model_manager", mock_model_manager):
            with patch("app.main.reload_task", AsyncMock()):
                from app.main import app

                client = TestClient(app, raise_server_exceptions=False)
                response = client.get("/model-info")

        assert response.status_code == 503


class TestReloadEndpoint:
    """Tests for /admin/reload endpoint."""

    def test_reload_success(self, mock_model_manager):
        """Test successful model reload."""
        mock_model_manager.load_champion = AsyncMock(return_value=True)

        with patch("app.main.model_manager", mock_model_manager):
            with patch("app.main.reload_task", AsyncMock()):
                from app.main import app

                client = TestClient(app, raise_server_exceptions=False)
                response = client.post("/admin/reload")

        assert response.status_code == 200
        data = response.json()
        assert data["reloaded"] is True
        assert "version" in data

    def test_reload_no_update(self, mock_model_manager):
        """Test reload when model hasn't changed."""
        mock_model_manager.load_champion = AsyncMock(return_value=False)

        with patch("app.main.model_manager", mock_model_manager):
            with patch("app.main.reload_task", AsyncMock()):
                from app.main import app

                client = TestClient(app, raise_server_exceptions=False)
                response = client.post("/admin/reload")

        assert response.status_code == 200
        data = response.json()
        assert data["reloaded"] is False

    def test_reload_champion_not_found(self, mock_model_manager):
        """Test reload when no champion model exists."""
        mock_model_manager.load_champion = AsyncMock(
            side_effect=ChampionNotFoundError("No champion model")
        )

        with patch("app.main.model_manager", mock_model_manager):
            with patch("app.main.reload_task", AsyncMock()):
                from app.main import app

                client = TestClient(app, raise_server_exceptions=False)
                response = client.post("/admin/reload")

        assert response.status_code == 404


class TestOpenAPISchema:
    """Tests for OpenAPI documentation."""

    def test_openapi_schema_available(self, mock_model_manager):
        """Test OpenAPI schema is accessible."""
        with patch("app.main.model_manager", mock_model_manager):
            with patch("app.main.reload_task", AsyncMock()):
                from app.main import app

                client = TestClient(app, raise_server_exceptions=False)
                response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema

    def test_docs_endpoint(self, mock_model_manager):
        """Test Swagger docs are available."""
        with patch("app.main.model_manager", mock_model_manager):
            with patch("app.main.reload_task", AsyncMock()):
                from app.main import app

                client = TestClient(app, raise_server_exceptions=False)
                response = client.get("/docs")

        assert response.status_code == 200
