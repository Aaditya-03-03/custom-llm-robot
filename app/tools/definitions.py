"""
Stage 7 Tool Definitions and Availability Registry for IOFT Humanoid Robot.
Defines which capabilities are physically verified and enabled for LLM tool calling.
"""

from typing import List, Set
from app.core.config import settings

ENABLED_STAGE_7_TOOLS: Set[str] = {
    "forward",
    "backward",
    "stop",
}

DISABLED_STAGE_7_TOOLS: Set[str] = {
    "left",
    "right",
    "stand",
    "sit",
    "status",
    "calibrate",
    "manual",
}


def is_tool_enabled(tool_name: str) -> bool:
    """
    Check if a tool is enabled for Stage 7 tool calling.
    left/right are only enabled if ROBOT_PIVOT_TURNS_VERIFIED is True.
    All other unverified capabilities (stand, sit, status, calibrate, manual) are disabled.
    """
    if not tool_name or not isinstance(tool_name, str):
        return False

    name = tool_name.lower().strip()
    if name in ENABLED_STAGE_7_TOOLS:
        return True

    if name in ("left", "right") and settings.ROBOT_PIVOT_TURNS_VERIFIED:
        return True

    return False


def get_enabled_tools() -> List[str]:
    """Return a list of all currently enabled tool names."""
    enabled = list(ENABLED_STAGE_7_TOOLS)
    if settings.ROBOT_PIVOT_TURNS_VERIFIED:
        enabled.extend(["left", "right"])
    return sorted(enabled)
