"""
Pydantic schemas for robot commands, strict parameter validation, and dry-run simulation.
"""

from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Strict parameter validation schemas (Strict types, no quiet coercion, extra forbidden)
# ---------------------------------------------------------------------------

class StrictMovementParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    speed: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Movement speed percentage strictly between 1 and 100",
    )
    steps: Optional[int] = Field(
        default=None,
        gt=0,
        description="Positive integer step count",
    )


class StrictStopParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    emergency: bool = Field(
        default=False,
        description="Emergency stop flag (strictly boolean)",
    )


class StrictEmptyParameters(BaseModel):
    """Schema for commands that forbid any parameters (stand, sit, status, calibrate, manual)."""
    model_config = ConfigDict(extra="forbid", strict=True)


# ---------------------------------------------------------------------------
# API Request / Response schemas for /commands/validate and /commands/dry-run
# ---------------------------------------------------------------------------

class CommandValidationRequest(BaseModel):
    tool_name: str = Field(..., description="Name of registered robot tool to validate")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool execution parameters to be strictly validated",
    )


class CommandValidationResponse(BaseModel):
    valid: bool = Field(..., description="True if command passes capability and safety checks")
    tool: str = Field(..., description="Canonical tool name")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Validated canonical parameters"
    )
    message: str = Field(..., description="Validation explanation or status summary")
    errors: List[str] = Field(
        default_factory=list, description="Specific error details if validation fails"
    )


class DryRunRequest(BaseModel):
    tool_name: str = Field(..., description="Name of registered robot tool to simulate")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Command parameters for dry-run simulation"
    )


class DryRunResponse(BaseModel):
    success: bool = Field(..., description="True if command validated and simulation completed")
    execution_mode: Literal["simulation"] = Field(
        default="simulation",
        description="Execution mode — strictly 'simulation' in Stage 5",
    )
    tool: str = Field(..., description="Canonical tool name")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Validated canonical parameters"
    )
    message: str = Field(..., description="Summary message of simulated execution")
    hardware_state_affected: bool = Field(
        default=False,
        description="Hardware safety invariant: strictly False in Stage 5",
    )


# ---------------------------------------------------------------------------
# Stage 6 Physical Locomotion Execution & Authority Schemas
# ---------------------------------------------------------------------------

from enum import Enum


class ExecutionStatus(str, Enum):
    EXECUTION_SUCCESS = "EXECUTION_SUCCESS"
    ACK_TIMEOUT = "ACK_TIMEOUT"
    NETWORK_ERROR = "NETWORK_ERROR"
    ROBOT_UNREACHABLE = "ROBOT_UNREACHABLE"
    INVALID_COMMAND = "INVALID_COMMAND"
    OWNERSHIP_REJECTED = "OWNERSHIP_REJECTED"
    EXECUTION_BUSY = "EXECUTION_BUSY"


class PhysicalExecutionRequest(BaseModel):
    tool_name: str = Field(..., description="Name of registered robot tool to physically execute")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution parameters. Movement requires explicit speed.",
    )


class PhysicalExecutionResponse(BaseModel):
    success: bool = Field(..., description="True if command validated and physically acknowledged")
    status: ExecutionStatus = Field(..., description="Detailed execution status")
    execution_id: str = Field(..., description="Unique UUID for execution audit tracking")
    tool: str = Field(..., description="Canonical tool name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Validated parameters executed")
    packet_sent: Optional[str] = Field(default=None, description="Exact UDP packet sent to ESP32")
    ack_received: bool = Field(default=False, description="True if MOTOR_ACK received from ESP32")
    hardware_state_affected: bool = Field(..., description="True once packet has been handed to socket")
    duration_seconds: Optional[float] = Field(default=None, description="Locomotion duration in seconds")
    message: str = Field(..., description="Human-readable outcome description")
    errors: List[str] = Field(default_factory=list, description="Specific error details if any")


class AuthorityStatusResponse(BaseModel):
    current_authority: str = Field(..., description="Current AI server execution authority: MANUAL or AI")
    changed_by: str = Field(..., description="Entity or client that last changed authority")
    changed_at: str = Field(..., description="ISO 8601 timestamp of last authority transition")


class AuthorityUpdateRequest(BaseModel):
    authority: str = Field(..., description="Target authority: 'MANUAL' or 'AI'")


# ---------------------------------------------------------------------------
# Preserved Stage 1 skeleton schemas for backward compatibility
# ---------------------------------------------------------------------------

class CommandExecutionRequest(BaseModel):
    tool_name: str = Field(..., description="Name of registered robot tool")
    parameters: dict = Field(default_factory=dict, description="Tool execution parameters")


class CommandExecutionResponse(BaseModel):
    success: bool
    message: str
    result: dict = Field(default_factory=dict)
