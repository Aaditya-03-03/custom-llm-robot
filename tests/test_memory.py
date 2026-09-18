"""
Stage 3 Memory Tests — ConversationRepository, MessageRepository, ConversationManager.

All tests use the isolated `custom_llm_robot_test` database.
These tests NEVER touch `custom_llm_robot`.

Run with:
    pytest tests/test_memory.py -v

Requires a running MongoDB instance at MONGODB_URI.
Tests are skipped automatically if MongoDB is unreachable.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from app.core.config import settings
from app.memory.repository import ConversationRepository, MessageRepository
from app.memory.manager import ConversationManager
from app.memory.database import is_mongodb_available


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_DB_NAME = "custom_llm_robot_test"

@pytest.fixture(scope="module")
def mongo_client():
    """
    Create a MongoClient for the isolated test database.
    Skip the entire module if MongoDB is unreachable.
    """
    try:
        client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        yield client
        # Teardown: drop the entire test database after the module finishes.
        client.drop_database(TEST_DB_NAME)
        client.close()
    except (ConnectionFailure, ServerSelectionTimeoutError):
        pytest.skip("MongoDB is not reachable — skipping memory tests.")


@pytest.fixture
def test_db(mongo_client):
    """Return the isolated test database and clear collections before each test."""
    db = mongo_client[TEST_DB_NAME]
    db["conversations"].delete_many({})
    db["messages"].delete_many({})
    return db


@pytest.fixture
def conv_repo(mongo_client, test_db):
    """ConversationRepository pointed at the test database."""
    with patch.object(settings, "MONGODB_DATABASE", TEST_DB_NAME):
        return ConversationRepository(mongo_client)


@pytest.fixture
def msg_repo(mongo_client, test_db):
    """MessageRepository pointed at the test database."""
    with patch.object(settings, "MONGODB_DATABASE", TEST_DB_NAME):
        return MessageRepository(mongo_client)


@pytest.fixture
def manager(mongo_client, test_db):
    """ConversationManager pointed at the test database."""
    with patch.object(settings, "MONGODB_DATABASE", TEST_DB_NAME):
        return ConversationManager(mongo_client)


# ---------------------------------------------------------------------------
# Test 1: MongoDB connection and index creation
# ---------------------------------------------------------------------------

def test_mongodb_connection_and_indexes(mongo_client):
    """Verify MongoDB is reachable and indexes can be created on the test database."""
    from pymongo import ASCENDING
    db = mongo_client[TEST_DB_NAME]

    db["conversations"].create_index(
        [("session_id", ASCENDING)], unique=True, name="test_idx_conv"
    )
    db["messages"].create_index(
        [("session_id", ASCENDING), ("timestamp", ASCENDING)],
        name="test_idx_msg"
    )
    index_names = [idx["name"] for idx in db["conversations"].list_indexes()]
    assert "test_idx_conv" in index_names

    msg_index_names = [idx["name"] for idx in db["messages"].list_indexes()]
    assert "test_idx_msg" in msg_index_names


# ---------------------------------------------------------------------------
# Test 2: auto-generate session_id when omitted
# ---------------------------------------------------------------------------

def test_chat_auto_session_id(manager):
    """Manager auto-generates a UUID session_id when None is passed."""
    session_id = manager.resolve_session(None)
    assert session_id is not None
    assert len(session_id) > 0
    # UUID format: 36 chars including hyphens
    assert len(session_id) == 36


# ---------------------------------------------------------------------------
# Test 3: explicit session_id reuse
# ---------------------------------------------------------------------------

def test_chat_explicit_session_id(manager):
    """Providing the same session_id twice reuses the session (no duplicate error)."""
    session_id = "test-explicit-session"
    first = manager.resolve_session(session_id)
    second = manager.resolve_session(session_id)
    assert first == session_id
    assert second == session_id


# ---------------------------------------------------------------------------
# Test 4: empty / whitespace session_id returns 400 (API layer)
# ---------------------------------------------------------------------------

def test_empty_session_id_returns_400():
    """
    Empty or whitespace-only session_id must return HTTP 400.
    Validated at the API layer — tested here via the FastAPI test client.
    """
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # Empty string
    response = client.post("/api/v1/chat", json={"message": "hello", "session_id": ""})
    assert response.status_code == 400, (
        f"Expected 400 for empty session_id, got {response.status_code}"
    )

    # Whitespace only
    response = client.post("/api/v1/chat", json={"message": "hello", "session_id": "   "})
    assert response.status_code == 400, (
        f"Expected 400 for whitespace session_id, got {response.status_code}"
    )


# ---------------------------------------------------------------------------
# Test 5: unknown session_id creates a new session
# ---------------------------------------------------------------------------

def test_unknown_session_id_creates_session(manager):
    """
    A valid but non-existent session_id must create the session, not error.
    """
    novel_id = "brand-new-session-xyz-999"
    session_id = manager.resolve_session(novel_id)
    assert session_id == novel_id


# ---------------------------------------------------------------------------
# Test 6: message persistence
# ---------------------------------------------------------------------------

def test_message_persistence(manager, msg_repo):
    """User and assistant messages are stored in MongoDB with UTC timestamps."""
    session_id = manager.resolve_session("persist-test")

    cutoff = datetime.now(timezone.utc)
    manager.save_user_message(session_id, "Hello robot!")
    manager.save_assistant_message(session_id, "Hello! How can I help?")

    _, messages = manager.get_full_history(session_id)
    assert len(messages) == 2

    user_msg = messages[0]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == "Hello robot!"
    assert user_msg["timestamp"].tzinfo is not None  # timezone-aware

    asst_msg = messages[1]
    assert asst_msg["role"] == "assistant"
    assert asst_msg["content"] == "Hello! How can I help?"
    assert asst_msg["timestamp"].tzinfo is not None


# ---------------------------------------------------------------------------
# Test 7: history limit
# ---------------------------------------------------------------------------

def test_history_limit(manager):
    """fetch_previous_history respects CONVERSATION_HISTORY_LIMIT and excludes current message."""
    session_id = manager.resolve_session("limit-test")
    limit = settings.CONVERSATION_HISTORY_LIMIT  # 10

    # Insert more messages than the limit
    for i in range(limit + 5):
        manager.save_user_message(session_id, f"Message {i}")

    cutoff = datetime.now(timezone.utc)
    history = manager.fetch_previous_history(session_id, before=cutoff)

    assert len(history) == limit, (
        f"Expected {limit} messages, got {len(history)}"
    )


# ---------------------------------------------------------------------------
# Test 8: history ordering — current message excluded
# ---------------------------------------------------------------------------

def test_current_message_not_in_history(manager):
    """
    The current user message must NOT appear in the fetched previous history.
    fetch_previous_history uses a cutoff timestamp captured BEFORE save_user_message.
    """
    session_id = manager.resolve_session("ordering-test")

    manager.save_user_message(session_id, "Old message")

    # Capture cutoff BEFORE saving current message
    cutoff = datetime.now(timezone.utc)
    history = manager.fetch_previous_history(session_id, before=cutoff)

    # Now save current message
    manager.save_user_message(session_id, "Current message")

    # History should only contain "Old message"
    assert len(history) == 1
    assert history[0]["content"] == "Old message"


# ---------------------------------------------------------------------------
# Test 9: GET and DELETE session history
# ---------------------------------------------------------------------------

def test_get_and_delete_session_history(manager):
    """GET history returns all messages; DELETE removes session and messages."""
    session_id = manager.resolve_session("delete-test")
    manager.save_user_message(session_id, "Question?")
    manager.save_assistant_message(session_id, "Answer!")

    exists, messages = manager.get_full_history(session_id)
    assert exists is True
    assert len(messages) == 2

    deleted = manager.delete_session(session_id)
    assert deleted is True

    exists_after, messages_after = manager.get_full_history(session_id)
    assert exists_after is False
    assert len(messages_after) == 0


# ---------------------------------------------------------------------------
# Test 10: follow-up conversation (context dependency)
# ---------------------------------------------------------------------------

def test_followup_conversation(manager):
    """
    Multi-turn history is stored and retrievable in order for follow-up questions.
    """
    session_id = manager.resolve_session("followup-test")

    manager.save_user_message(session_id, "What platform is used for human-robot interaction?")
    manager.save_assistant_message(session_id, "The robot uses a Meta Quest headset.")

    cutoff = datetime.now(timezone.utc)
    history = manager.fetch_previous_history(session_id, before=cutoff)

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert "Meta Quest" in history[1]["content"]


# ---------------------------------------------------------------------------
# Test 11: MongoDB unavailable returns HTTP 503 on chat
# ---------------------------------------------------------------------------

def test_mongodb_unresponsive_error_handling():
    """
    When MongoDB is unavailable, POST /api/v1/chat returns HTTP 503.
    No silent fallback. No mock LLM substitution.
    """
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    with patch("app.api.chat.is_mongodb_available", return_value=False):
        response = client.post(
            "/api/v1/chat",
            json={"message": "Hello!"},
        )
    assert response.status_code == 503
    data = response.json()
    assert "MongoDB" in data["detail"] or "unavailable" in data["detail"].lower()
