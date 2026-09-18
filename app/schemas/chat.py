"""
Pydantic schemas for chat requests and responses.
"""

from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., description="User instruction or prompt")

class ChatResponse(BaseModel):
    session_id: str
    response: str
    actions_taken: list[dict] = Field(default_factory=list)
