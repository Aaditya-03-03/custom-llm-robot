"""
Pydantic schemas and enums for Intent Detection and Structured Commands.
"""

from enum import Enum
from typing import Optional, Union, Dict, Any
from pydantic import BaseModel, Field


class IntentCategory(str, Enum):
    CONVERSATION = "conversation"    # Questions, chitchat, RAG knowledge queries
    ROBOT_COMMAND = "robot_command"  # Valid recognized robot commands
    UNSUPPORTED = "unsupported"      # Unsupported, invalid, or out-of-scope requests


class RobotAction(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    STAND = "stand"
    SIT = "sit"
    STOP = "stop"
    STATUS = "status"
    CALIBRATE = "calibrate"
    MANUAL = "manual"


class MovementParameters(BaseModel):
    speed: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Movement speed percentage from 1 to 100",
    )
    steps: Optional[int] = Field(
        default=None,
        gt=0,
        description="Number of steps to take (must be positive)",
    )


class StopParameters(BaseModel):
    emergency: bool = Field(
        default=False,
        description="Emergency stop flag (True if emergency stop was explicitly stated)",
    )


class StructuredIntent(BaseModel):
    category: IntentCategory = Field(..., description="High-level category of user message")
    action: Optional[RobotAction] = Field(
        default=None,
        description="Canonical robot action if category is robot_command",
    )
    parameters: Optional[Union[MovementParameters, StopParameters, Dict[str, Any]]] = Field(
        default=None,
        description="Extracted command parameters",
    )
    raw_input: str = Field(..., description="Original user text")
    is_valid: bool = Field(
        default=True,
        description="True if structured intent was successfully validated",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error details if validation failed or command is unsupported",
    )


class IntentRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Natural language instruction to be parsed for robot intent",
        json_schema_extra={"example": "Move forward 3 steps at speed 50"},
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for tracking context",
    )


class IntentAuditRecord(BaseModel):
    """
    In-memory audit log record. Decoupled from runtime DB persistence.
    Reserved for developer review to curate future Stage 7 fine-tuning datasets.
    """
    timestamp: str
    raw_input: str
    detected_category: str
    detected_action: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    is_valid: bool
