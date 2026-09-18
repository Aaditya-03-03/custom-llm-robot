"""
Chat endpoint tests — Stage 1 tests preserved + Stage 3 session memory tests.
"""
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

client = TestClient(app)


# ---------------------------------------------------------------------------
# Stage 1 tests — must continue passing unchanged
# ---------------------------------------------------------------------------

def test_chat_mock_provider(monkeypatch):
    """Basic chat returns 200 with session_id and response (MongoDB mocked)."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")

    with patch("app.api.chat.is_mongodb_available", return_value=True), \
         patch("app.api.chat.ConversationManager") as MockMgr, \
         patch("app.api.chat.VectorStoreManager") as MockVS:

        mock_mgr_instance = MockMgr.return_value
        mock_mgr_instance.resolve_session.return_value = "test-session-001"
        mock_mgr_instance.fetch_previous_history.return_value = []
        mock_mgr_instance.save_user_message.return_value = None
        mock_mgr_instance.save_assistant_message.return_value = None

        mock_vs_instance = MockVS.return_value
        mock_vs_instance.query.return_value = []

        payload = {"message": "Hello, who are you?"}
        response = client.post("/api/v1/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "session_id" in data
    assert "IOFT Humanoid Robot AI" in data["response"]
    assert "X-Request-ID" in response.headers


def test_chat_invalid_payload():
    """Empty payload returns 422 Unprocessable Entity."""
    response = client.post("/api/v1/chat", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Stage 3 tests — session memory behaviour
# ---------------------------------------------------------------------------

def test_chat_returns_session_id():
    """POST /api/v1/chat always returns a session_id in the response."""
    with patch("app.api.chat.is_mongodb_available", return_value=True), \
         patch("app.api.chat.ConversationManager") as MockMgr, \
         patch("app.api.chat.VectorStoreManager") as MockVS, \
         patch("app.api.chat.get_llm_provider") as MockLLM:

        mock_mgr_instance = MockMgr.return_value
        mock_mgr_instance.resolve_session.return_value = "auto-uuid-session"
        mock_mgr_instance.fetch_previous_history.return_value = []
        mock_mgr_instance.save_user_message.return_value = None
        mock_mgr_instance.save_assistant_message.return_value = None

        mock_vs_instance = MockVS.return_value
        mock_vs_instance.query.return_value = []

        mock_provider = AsyncMock()
        mock_provider.generate_response = AsyncMock(return_value="I am the robot assistant.")
        MockLLM.return_value = mock_provider

        response = client.post("/api/v1/chat", json={"message": "Hello!"})

    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["session_id"] == "auto-uuid-session"


def test_chat_empty_session_id_returns_400():
    """Empty session_id returns 400 without touching MongoDB."""
    response = client.post(
        "/api/v1/chat",
        json={"message": "Hello!", "session_id": ""}
    )
    assert response.status_code == 400


def test_chat_whitespace_session_id_returns_400():
    """Whitespace-only session_id returns 400."""
    response = client.post(
        "/api/v1/chat",
        json={"message": "Hello!", "session_id": "   "}
    )
    assert response.status_code == 400


def test_chat_mongodb_unavailable_returns_503():
    """When MongoDB is offline, POST /api/v1/chat returns 503."""
    with patch("app.api.chat.is_mongodb_available", return_value=False):
        response = client.post(
            "/api/v1/chat",
            json={"message": "Hello!"}
        )
    assert response.status_code == 503
