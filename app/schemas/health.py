"""
Pydantic schemas for health check responses.
"""
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="General server status")
    llm_provider: str = Field(..., description="Active LLM provider backend")
    model: str = Field(..., description="Configured LLM model name")
    llm_available: bool = Field(
        ..., description="Whether the local LLM runtime is online and responsive"
    )
    mongodb_available: bool = Field(
        ..., description="Whether the MongoDB conversation memory store is reachable"
    )
    robot_connected: bool = Field(
        default=False, description="Whether a fresh successful heartbeat confirms active robot connectivity"
    )
    robot_state_available: bool = Field(
        default=True, description="Whether the canonical RobotState subsystem is initialized and usable"
    )
