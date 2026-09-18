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
# Preserved Stage 1 skeleton schemas for backward compatibility
# ---------------------------------------------------------------------------

class CommandExecutionRequest(BaseModel):
    tool_name: str = Field(..., description="Name of registered robot tool")
    parameters: dict = Field(default_factory=dict, description="Tool execution parameters")


class CommandExecutionResponse(BaseModel):
    success: bool
    message: str
    result: dict = Field(default_factory=dict)
