"""
Health endpoint tests — Stage 3 extended to verify mongodb_available field.
"""
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Health endpoint returns expected fields including mongodb_available."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "llm_provider" in data
    assert "model" in data
    assert "llm_available" in data
    assert "mongodb_available" in data  # Stage 3 addition
    assert isinstance(data["mongodb_available"], bool)
    assert "X-Request-ID" in response.headers


def test_health_reports_mongodb_unavailable():
    """When MongoDB is down, health reports mongodb_available=false (not an error)."""
    with patch("app.api.health.is_mongodb_available", return_value=False):
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["mongodb_available"] is False
