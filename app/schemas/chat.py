"""
Pydantic schemas for chat requests and responses.
"""
from typing import List, Literal, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        ..., description="Role of the message author"
    )
    content: str = Field(..., description="Content of the message")


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="User instruction or chat message",
        json_schema_extra={"example": "What platform is used for human-robot interaction?"},
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional session identifier. If omitted, a new UUID session is created. "
            "If provided but empty/whitespace, returns HTTP 400."
        ),
        json_schema_extra={"example": "robot-demo-001"},
    )


class ChatResponse(BaseModel):
    session_id: str = Field(..., description="Session identifier (auto-generated or provided)")
    response: str = Field(..., description="Generated text response from local LLM")


class MessageRecord(BaseModel):
    """A single message record as returned by the history endpoint."""
    role: Literal["user", "assistant"] = Field(..., description="Message author role")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="UTC timestamp of when message was stored")


class HistoryResponse(BaseModel):
    """Response schema for GET /api/v1/chat/{session_id}/history."""
    session_id: str = Field(..., description="Session identifier")
    messages: List[MessageRecord] = Field(
        default_factory=list, description="Full message history for this session, ordered oldest first"
    )
