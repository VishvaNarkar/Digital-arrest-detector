"""Unit tests for the FastAPI backend API endpoints.

These tests run without loading full machine learning model files (they are stubbed out)
to run fast and run successfully in any environment.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Mock out model loading so it doesn't load the real files in memory during tests
# or fail if keras files are not present in CI.
with patch("backend.core.models.load_all") as mock_load:
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.core import models as backend_models

    # Manually assign mock objects to singletons
    backend_models.text_model = MagicMock()
    backend_models.vectorizer = MagicMock()
    backend_models.deepfake_model = MagicMock()

    # Configure predict_proba mock return value
    # Format of predict_proba output is array of shape (n_samples, n_classes)
    backend_models.text_model.predict_proba.return_value = [[0.9, 0.1]]

client = TestClient(app)


def test_health_endpoint():
    """Verify health endpoint returns status of loaded models."""
    response = client.get("/api/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["ok"] is True
    assert "ollama_available" in json_data
    assert json_data["text_model_loaded"] is True
    assert json_data["deepfake_model_loaded"] is True


def test_text_analysis_endpoint_safe():
    """Verify text analysis endpoint classifies normal message as safe."""
    # Mock ML probability output to be very low (safe)
    backend_models.text_model.predict_proba.return_value = [[0.99, 0.01]]

    response = client.post(
        "/api/analyze/text",
        json={"text": "Hello, how are you?", "threshold": 0.35},
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "Safe" in json_data["label"]
    assert json_data["ml_prob"] == 0.01
    assert json_data["keyword_score"] == 0
    assert len(json_data["keywords"]) == 0
    assert "compound" in json_data["sentiment"]


def test_text_analysis_endpoint_scam():
    """Verify text analysis endpoint detects scam with keyword match."""
    # Mock ML probability output to be high (scam)
    backend_models.text_model.predict_proba.return_value = [[0.1, 0.9]]

    response = client.post(
        "/api/analyze/text",
        json={"text": "URGENT: Verify your account and details immediately!", "threshold": 0.35},
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "Scam" in json_data["label"]
    assert json_data["keyword_score"] > 0
    assert len(json_data["keywords"]) > 0
    assert "verify details" in json_data["keywords"] or "urgent" in json_data["keywords"]


def test_text_analysis_empty_body():
    """Verify endpoint rejects empty or whitespace-only text."""
    response = client.post(
        "/api/analyze/text",
        json={"text": "   ", "threshold": 0.35},
    )
    assert response.status_code == 422
