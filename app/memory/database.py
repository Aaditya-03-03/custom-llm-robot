"""
MongoDB client singleton, connection lifecycle management, index creation,
and availability health check.

Design rules (per implementation plan):
- The FastAPI application may start even when MongoDB is unavailable.
- POST /api/v1/chat returns HTTP 503 while MongoDB is unreachable.
- No silent fallback to another database.
- All timestamps are UTC timezone-aware.
"""

import logging
from typing import Optional

from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from app.core.config import settings

logger = logging.getLogger("custom_llm_robot.memory.database")

# Module-level singleton — set during FastAPI lifespan startup.
_mongo_client: Optional[MongoClient] = None


def get_mongo_client() -> Optional[MongoClient]:
    """Return the active MongoClient singleton, or None if not yet initialised."""
    return _mongo_client


def is_mongodb_available() -> bool:
    """
    Perform a lightweight ping against the MongoDB server.
    Returns True if the server responds, False otherwise.
    Does NOT raise — safe to call from health checks.
    """
    client = _mongo_client
    if client is None:
        return False
    try:
        client.admin.command("ping")
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.warning(f"MongoDB ping failed: {e}")
        return False
    except Exception as e:
        logger.warning(f"MongoDB availability check error: {e}")
        return False


def connect_mongodb() -> None:
    """
    Initialise the MongoClient singleton and create required indexes.

    Called once during FastAPI lifespan startup.
    If the MongoDB server is unreachable at startup, the client is still
    created (connection is lazy); the server-selection timeout is kept
    short so health checks respond quickly.
    """
    global _mongo_client
    logger.info(f"Connecting to MongoDB at {settings.MONGODB_URI} ...")

    # serverSelectionTimeoutMS: how long a command waits before giving up.
    # Kept short (3 s) so health checks are non-blocking.
    _mongo_client = MongoClient(
        settings.MONGODB_URI,
        serverSelectionTimeoutMS=3000,
    )

    # Attempt index creation; if MongoDB is offline this will silently fail
    # and the app continues — 503 will be returned on chat requests instead.
    try:
        _ensure_indexes(_mongo_client)
        logger.info("MongoDB indexes verified/created successfully.")
    except Exception as e:
        logger.warning(
            f"MongoDB index creation skipped (server may be offline): {e}"
        )


def disconnect_mongodb() -> None:
    """Close the MongoClient singleton. Called during FastAPI lifespan shutdown."""
    global _mongo_client
    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None
        logger.info("MongoDB connection closed.")


def _ensure_indexes(client: MongoClient) -> None:
    """
    Create idempotent indexes on conversations and messages collections.

    conversations:
        unique index on session_id

    messages:
        compound index on (session_id ASC, timestamp ASC)
        — used for history retrieval ordering.
    """
    db = client[settings.MONGODB_DATABASE]

    conversations = db[settings.MONGODB_CONVERSATIONS_COLLECTION]
    conversations.create_index(
        [("session_id", ASCENDING)],
        unique=True,
        name="idx_conversations_session_id",
    )

    messages = db[settings.MONGODB_MESSAGES_COLLECTION]
    messages.create_index(
        [("session_id", ASCENDING), ("timestamp", ASCENDING)],
        name="idx_messages_session_timestamp",
    )
