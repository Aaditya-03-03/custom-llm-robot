"""
Unit tests for Stage 8 RobotState Models and Enum definitions.
Verifies initial states, enum memberships, and standardized fields.
"""

from datetime import datetime, timezone
import pytest

from app.state.models import (
    RobotState,
    ConnectionStatus,
    MotionState,
    ExecutionState,
    WatchdogStatus,
)
from app.core.config import ControlAuthority


def test_robot_state_initial_defaults():
    """Verify clean initial state: UNKNOWN/DISCONNECTED, no false CONNECTED."""
    state = RobotState()
    assert state.connection_status == ConnectionStatus.UNKNOWN
    assert state.authority == ControlAuthority.MANUAL
    assert state.execution_status == ExecutionState.IDLE
    assert state.motion_state == MotionState.UNKNOWN
    assert state.watchdog_status == WatchdogStatus.UNKNOWN
    assert state.is_stale is False
    assert state.requested_command is None
    assert state.requested_speed is None
    assert state.current_command is None
    assert state.last_command is None
    assert state.last_ack is None
    assert state.last_ack_timestamp is None
    assert state.last_heartbeat_timestamp is None


def test_robot_state_standardized_fields():
    """Verify distinct command state fields vs physical observed state."""
    now = datetime.now(timezone.utc)
    state = RobotState(
        connection_status=ConnectionStatus.CONNECTED,
        authority=ControlAuthority.AI,
        requested_command="FORWARD",
        requested_speed=20,
        current_command="FORWARD",
        last_command="STOP",
        execution_status=ExecutionState.EXECUTING,
        motion_state=MotionState.UNKNOWN,
        last_ack="MOTOR_ACK",
        last_ack_timestamp=now,
        last_heartbeat_timestamp=now,
        watchdog_status=WatchdogStatus.HEALTHY,
        is_stale=False,
    )
    assert state.requested_command == "FORWARD"
    assert state.requested_speed == 20
    assert state.current_command == "FORWARD"
    assert state.last_command == "STOP"
    assert state.motion_state == MotionState.UNKNOWN
    assert state.last_ack == "MOTOR_ACK"
    assert state.watchdog_status == WatchdogStatus.HEALTHY
