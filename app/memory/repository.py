"""
PyMongo CRUD repository for conversations and messages collections.

All methods are synchronous (PyMongo sync driver).
All timestamps stored are UTC timezone-aware datetimes.
History is always returned sorted by (timestamp ASC, _id ASC) for
deterministic ordering even when timestamps collide.

Note on PyMongo timezone handling:
    PyMongo by default returns datetime objects WITHOUT tzinfo (naive UTC),
    even though the values stored are in UTC. We attach timezone.utc on
    read so all timestamps returned from this repository are timezone-aware,
    matching the UTC contract stated in the implementation plan.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError

from app.core.config import settings
from app.memory.models import ConversationDocument, MessageDocument


def _make_aware(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure the 'timestamp' field in a message document is UTC timezone-aware.

    PyMongo returns naive datetimes (no tzinfo) even when the stored values
    are in UTC. This function attaches timezone.utc to any naive timestamp
    so callers always receive timezone-aware datetimes.
    """
    ts = doc.get("timestamp")
    if isinstance(ts, datetime) and ts.tzinfo is None:
        doc["timestamp"] = ts.replace(tzinfo=timezone.utc)
    return doc

logger = logging.getLogger("custom_llm_robot.memory.repository")


class ConversationRepository:
    """
    Manages the `conversations` collection.

    Responsibilities:
    - Create a session document (upsert-safe; no error if session already exists).
    - Check session existence.
    - Update session updated_at timestamp.
    - Delete a session document.
    """

    def __init__(self, client: MongoClient) -> None:
        db = client[settings.MONGODB_DATABASE]
        self._col = db[settings.MONGODB_CONVERSATIONS_COLLECTION]

    def get_or_create(self, session_id: str) -> bool:
        """
        Ensure a conversation document exists for session_id.
        Returns True if the session was newly created, False if it already existed.
        """
        now = datetime.now(timezone.utc)
        try:
            self._col.insert_one(
                ConversationDocument(
                    session_id=session_id,
                    created_at=now,
                    updated_at=now,
                ).model_dump()
            )
            logger.info(f"Created new session: {session_id}")
            return True
        except DuplicateKeyError:
            # Session already exists — not an error.
            logger.debug(f"Session already exists: {session_id}")
            return False

    def exists(self, session_id: str) -> bool:
        """Return True if a conversation document exists for session_id."""
        return (
            self._col.count_documents({"session_id": session_id}, limit=1) > 0
        )

    def touch(self, session_id: str) -> None:
        """Update the updated_at timestamp of the session to now (UTC)."""
        self._col.update_one(
            {"session_id": session_id},
            {"$set": {"updated_at": datetime.now(timezone.utc)}},
        )

    def delete(self, session_id: str) -> bool:
        """
        Delete the conversation document for session_id.
        Returns True if a document was deleted, False if it did not exist.
        """
        result = self._col.delete_one({"session_id": session_id})
        deleted = result.deleted_count > 0
        if deleted:
            logger.info(f"Deleted conversation document: {session_id}")
        return deleted


class MessageRepository:
    """
    Manages the `messages` collection.

    Responsibilities:
    - Save a message (user or assistant).
    - Retrieve the N most recent messages for a session BEFORE a given cutoff time.
    - Retrieve all messages for a session (history endpoint).
    - Delete all messages for a session.
    """

    def __init__(self, client: MongoClient) -> None:
        db = client[settings.MONGODB_DATABASE]
        self._col = db[settings.MONGODB_MESSAGES_COLLECTION]

    def save(self, session_id: str, role: str, content: str) -> None:
        """
        Persist a message document with a UTC timestamp.
        The timestamp is set at call-time, so user messages always have an
        earlier timestamp than the assistant messages that follow them.
        """
        doc = MessageDocument(
            session_id=session_id,
            role=role,
            content=content,
        )
        self._col.insert_one(doc.model_dump())
        logger.debug(f"Saved {role} message for session {session_id}")

    def get_previous_history(
        self,
        session_id: str,
        limit: int,
        before: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Return up to `limit` messages for session_id that were stored
        STRICTLY BEFORE `before` (the moment the current user message
        is being saved).

        Sorted by (timestamp ASC, _id ASC) for deterministic ordering.
        Returns the LAST `limit` messages in chronological order.
        """
        cursor = (
            self._col.find(
                {
                    "session_id": session_id,
                    "timestamp": {"$lt": before},
                }
            )
            .sort([("timestamp", ASCENDING), ("_id", ASCENDING)])
        )
        all_docs = list(cursor)
        # Keep only the most recent N, then return in chronological order.
        recent = all_docs[-limit:] if len(all_docs) > limit else all_docs
        # Normalise naive timestamps → UTC-aware before returning.
        return [_make_aware(doc) for doc in recent]

    def get_all(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Return all messages for session_id sorted chronologically.
        Used by the history management endpoint.
        """
        cursor = self._col.find({"session_id": session_id}).sort(
            [("timestamp", ASCENDING), ("_id", ASCENDING)]
        )
        # Normalise naive timestamps → UTC-aware before returning.
        return [_make_aware(doc) for doc in cursor]

    def delete_all(self, session_id: str) -> int:
        """
        Delete all messages for session_id.
        Returns the count of deleted documents.
        """
        result = self._col.delete_many({"session_id": session_id})
        logger.info(
            f"Deleted {result.deleted_count} message(s) for session {session_id}"
        )
        return result.deleted_count
