"""
Pydantic document models for Conversation and Message MongoDB documents.

All timestamps are UTC timezone-aware datetimes (datetime.now(timezone.utc)).
Documents are never stored with naive datetimes or string timestamps.
"""

from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Return the current UTC datetime, timezone-aware."""
    return datetime.now(timezone.utc)


class ConversationDocument(BaseModel):
    """
    Represents a row in the `conversations` collection.

    session_id is the unique key. created_at and updated_at are both UTC.
    """
    session_id: str = Field(..., description="Unique session identifier")
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC datetime when the session was first created",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC datetime when the session was last modified",
    )


class MessageDocument(BaseModel):
    """
    Represents a row in the `messages` collection.

    session_id links this message to its conversation.
    timestamp is UTC and is the primary sort key for history retrieval.
    MongoDB _id serves as the secondary sort key for deterministic ordering
    when timestamps are equal.
    """
    session_id: str = Field(..., description="Session this message belongs to")
    role: Literal["user", "assistant"] = Field(
        ..., description="Message author role"
    )
    content: str = Field(..., description="Message text content")
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC datetime when this message was stored",
    )
