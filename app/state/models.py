"""
Pydantic Models and Enums for Stage 8 Robot State Awareness & Telemetry.
Standardizes representation of connection, authority, command state, and physical observed state.
"""

from enum import Enum
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from app.core.config import ControlAuthority


class ConnectionStatus(str, Enum):
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    UNKNOWN = "UNKNOWN"


class MotionState(str, Enum):
    STOPPED = "STOPPED"
    MOVING = "MOVING"
    UNKNOWN = "UNKNOWN"


class ExecutionState(str, Enum):
    IDLE = "IDLE"
    EXECUTING = "EXECUTING"
    BUSY = "BUSY"
    ERROR = "ERROR"


class WatchdogStatus(str, Enum):
    HEALTHY = "HEALTHY"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"


class RobotState(BaseModel):
    """
    Canonical, normalized snapshot of the robot's operational and telemetry state.
    Strictly separates requested command state from physical observed state.
    """
    connection_status: ConnectionStatus = Field(
        default=ConnectionStatus.UNKNOWN,
        description="Current connectivity to ESP32 (derived from active heartbeats)."
    )
    authority: ControlAuthority = Field(
        default=ControlAuthority.MANUAL,
        description="Current AI Server control authority (MANUAL or AI)."
    )

    # Command State (what the AI server requested)
    requested_command: Optional[str] = Field(
        default=None,
        description="Last command verb requested by the AI Server."
    )
    requested_speed: Optional[int] = Field(
        default=None,
        description="Speed percentage requested by the AI Server (1-100)."
    )

    # Observed / Execution State
    current_command: Optional[str] = Field(
        default=None,
        description="Active executing command, or 'STOP' if halted."
    )
    last_command: Optional[str] = Field(
        default=None,
        description="Previously executed command before the current state."
    )
    execution_status: ExecutionState = Field(
        default=ExecutionState.IDLE,
        description="Current execution lifecycle phase (IDLE, EXECUTING, BUSY, ERROR)."
    )
    motion_state: MotionState = Field(
        default=MotionState.UNKNOWN,
        description="Observed physical motion state. Remains UNKNOWN during execution unless verified by telemetry."
    )

    # Telemetry & Acknowledgment Timestamps
    last_ack: Optional[str] = Field(
        default=None,
        description="Payload of the last received command acknowledgment (e.g. 'MOTOR_ACK')."
    )
    last_ack_timestamp: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp of the last received command acknowledgment."
    )
    last_heartbeat_timestamp: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp of the last successful reachability probe (PONG)."
    )
    last_command_timestamp: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when the last command was dispatched."
    )
    last_telemetry_timestamp: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp of the last rich telemetry status packet."
    )

    # Watchdog & Freshness
    watchdog_status: WatchdogStatus = Field(
        default=WatchdogStatus.UNKNOWN,
        description="Reported ESP32 hardware watchdog status."
    )
    is_stale: bool = Field(
        default=False,
        description="True if time since last heartbeat exceeds ROBOT_STATE_STALE_AFTER_SECONDS."
    )
