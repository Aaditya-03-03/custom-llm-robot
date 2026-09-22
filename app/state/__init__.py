"""
Robot State and Telemetry package for Stage 8.
"""

from app.state.models import (
    RobotState,
    ConnectionStatus,
    MotionState,
    ExecutionState,
    WatchdogStatus,
)
from app.state.telemetry import (
    TelemetryParser,
    TelemetryType,
    ParsedTelemetry,
)
from app.state.manager import (
    RobotStateManager,
    default_state_manager,
)

__all__ = [
    "RobotState",
    "ConnectionStatus",
    "MotionState",
    "ExecutionState",
    "WatchdogStatus",
    "TelemetryParser",
    "TelemetryType",
    "ParsedTelemetry",
    "RobotStateManager",
    "default_state_manager",
]
