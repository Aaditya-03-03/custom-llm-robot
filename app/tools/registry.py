"""
Closed, immutable Tool Registry for IOFT Humanoid Robot.
Defines metadata allowlist and parameter schemas for grounded capabilities.

Architectural Boundaries & Division of Responsibility:
- ToolRegistry:
    identifies the approved tool
    exposes its parameter schema
    does NOT perform validation logic
    does NOT perform execution
- CommandSafetyValidator:
    performs the single authoritative validation pipeline
    fails closed on parameter/range/type violations
"""

from typing import Dict, List, Optional, Type
from pydantic import BaseModel, ConfigDict
from app.schemas.commands import (
    StrictMovementParameters,
    StrictStopParameters,
    StrictEmptyParameters,
)


class RegisteredTool(BaseModel):
    """
    Immutable tool definition representing an allowlisted robot capability.
    model_config frozen=True guarantees tool definitions cannot be mutated at runtime.
    """
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameter_schema: Optional[Type[BaseModel]] = None
    requires_parameters: bool = False


# Canonical grounded tools established in Stage 4
_GROUNDED_TOOLS: Dict[str, RegisteredTool] = {
    "forward": RegisteredTool(
        name="forward",
        description="Linear directional movement forward",
        parameter_schema=StrictMovementParameters,
        requires_parameters=False,
    ),
    "backward": RegisteredTool(
        name="backward",
        description="Linear directional movement backward",
        parameter_schema=StrictMovementParameters,
        requires_parameters=False,
    ),
    "left": RegisteredTool(
        name="left",
        description="Exact directional movement left (no rotational turn semantics)",
        parameter_schema=StrictMovementParameters,
        requires_parameters=False,
    ),
    "right": RegisteredTool(
        name="right",
        description="Exact directional movement right (no rotational turn semantics)",
        parameter_schema=StrictMovementParameters,
        requires_parameters=False,
    ),
    "stand": RegisteredTool(
        name="stand",
        description="Posture transition to standing stance",
        parameter_schema=StrictEmptyParameters,
        requires_parameters=False,
    ),
    "sit": RegisteredTool(
        name="sit",
        description="Posture transition to sitting / rest stance",
        parameter_schema=StrictEmptyParameters,
        requires_parameters=False,
    ),
    "stop": RegisteredTool(
        name="stop",
        description="Halt all robot movement immediately",
        parameter_schema=StrictStopParameters,
        requires_parameters=False,
    ),
    "status": RegisteredTool(
        name="status",
        description="Query robot status and telemetry (interpretation-only in Stage 5)",
        parameter_schema=StrictEmptyParameters,
        requires_parameters=False,
    ),
    "calibrate": RegisteredTool(
        name="calibrate",
        description="Zero sensors and joints (interpretation-only in Stage 5)",
        parameter_schema=StrictEmptyParameters,
        requires_parameters=False,
    ),
    "manual": RegisteredTool(
        name="manual",
        description="Switch to manual override mode (interpretation-only in Stage 5)",
        parameter_schema=StrictEmptyParameters,
        requires_parameters=False,
    ),
}


class ToolRegistry:
    """
    Closed, immutable allowlist registry.
    Resolves canonical capability names to frozen RegisteredTool definitions.
    Rejects any unverified, arbitrary, or user-supplied tool names.
    """

    def __init__(self):
        # Tools are pre-populated and fixed
        self._tools: Dict[str, RegisteredTool] = dict(_GROUNDED_TOOLS)

    def get_tool(self, name: str) -> Optional[RegisteredTool]:
        """Resolve a tool name against the closed allowlist."""
        if not name or not isinstance(name, str):
            return None
        return self._tools.get(name.lower().strip())

    def is_registered(self, name: str) -> bool:
        """Check if a tool name is allowlisted."""
        if not name or not isinstance(name, str):
            return False
        return name.lower().strip() in self._tools

    def list_tools(self) -> List[str]:
        """Return list of all allowlisted tool names."""
        return list(self._tools.keys())


# Default singleton instance
default_tool_registry = ToolRegistry()
