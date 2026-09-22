"""
Central Robot State Manager for IOFT Humanoid Robot.
Maintains canonical operational state, evaluates heartbeat freshness,
manages command/physical state separation, and performs independent heartbeat probes.
"""

import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

from app.core.config import settings, ControlAuthority
from app.state.models import (
    RobotState,
    ConnectionStatus,
    MotionState,
    ExecutionState,
    WatchdogStatus,
)
from app.state.telemetry import TelemetryParser, TelemetryType

logger = logging.getLogger("custom_llm_robot.state.manager")


class RobotStateManager:
    """
    Thread-safe canonical state manager maintaining the single source of truth
    for the robot's physical and operational state.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Initial state: UNKNOWN / DISCONNECTED (no false "CONNECTED" on startup)
        self._connection_status: ConnectionStatus = ConnectionStatus.UNKNOWN
        self._requested_command: Optional[str] = None
        self._requested_speed: Optional[int] = None
        self._current_command: Optional[str] = None
        self._last_command: Optional[str] = None
        self._execution_status: ExecutionState = ExecutionState.IDLE
        self._motion_state: MotionState = MotionState.UNKNOWN

        self._last_ack: Optional[str] = None
        self._last_ack_timestamp: Optional[datetime] = None
        self._last_heartbeat_timestamp: Optional[datetime] = None
        self._last_command_timestamp: Optional[datetime] = None
        self._last_telemetry_timestamp: Optional[datetime] = None

        self._watchdog_status: WatchdogStatus = WatchdogStatus.UNKNOWN

    def get_state(self) -> RobotState:
        """
        Produce a clean, point-in-time snapshot of the robot state.
        Calculates freshness on the fly using ROBOT_STATE_STALE_AFTER_SECONDS.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            is_stale = False
            conn_status = self._connection_status

            if self._connection_status == ConnectionStatus.DISCONNECTED:
                conn_status = ConnectionStatus.DISCONNECTED
                is_stale = True
            elif self._last_heartbeat_timestamp is not None:
                elapsed = (now - self._last_heartbeat_timestamp).total_seconds()
                if elapsed > settings.ROBOT_STATE_STALE_AFTER_SECONDS:
                    is_stale = True
                    conn_status = ConnectionStatus.DISCONNECTED
                else:
                    is_stale = False
                    conn_status = ConnectionStatus.CONNECTED

            from app.execution.authority import default_authority_manager
            authority = default_authority_manager.current_authority

            return RobotState(
                connection_status=conn_status,
                authority=authority,
                requested_command=self._requested_command,
                requested_speed=self._requested_speed,
                current_command=self._current_command,
                last_command=self._last_command,
                execution_status=self._execution_status,
                motion_state=self._motion_state,
                last_ack=self._last_ack,
                last_ack_timestamp=self._last_ack_timestamp,
                last_heartbeat_timestamp=self._last_heartbeat_timestamp,
                last_command_timestamp=self._last_command_timestamp,
                last_telemetry_timestamp=self._last_telemetry_timestamp,
                watchdog_status=self._watchdog_status,
                is_stale=is_stale,
            )

    def is_connected(self) -> bool:
        """Return True only if a fresh heartbeat confirms active connectivity."""
        state = self.get_state()
        return state.connection_status == ConnectionStatus.CONNECTED and not state.is_stale

    def record_command_start(self, tool_name: str, speed: Optional[int] = None) -> None:
        """
        Record that a command has been dispatched by the AI Server.
        Motion state remains UNKNOWN (ACK != physical movement).
        """
        with self._lock:
            cmd = tool_name.upper().strip()
            self._requested_command = cmd
            self._requested_speed = speed
            self._current_command = cmd
            self._execution_status = ExecutionState.EXECUTING
            self._last_command_timestamp = datetime.now(timezone.utc)
            # Motion state remains UNKNOWN or preserved
            if cmd == "STOP":
                self._requested_speed = 0
            logger.debug(f"State: Command start recorded: {cmd}, speed={speed}")

    def record_ack(self, ack_payload: str) -> None:
        """
        Record a command acknowledgment from the ESP32.
        Confirms command reception and reachability. Motion state remains UNKNOWN.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            self._last_ack = ack_payload
            self._last_ack_timestamp = now
            self._last_heartbeat_timestamp = now
            self._connection_status = ConnectionStatus.CONNECTED
            self._execution_status = ExecutionState.EXECUTING
            # Notice: motion_state is explicitly NOT set to MOVING!
            logger.debug(f"State: ACK recorded: '{ack_payload}'")

    def record_heartbeat(self) -> None:
        """
        Record a successful reachability heartbeat (PONG).
        Confirms connectivity without mutating motion state.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            self._last_heartbeat_timestamp = now
            self._connection_status = ConnectionStatus.CONNECTED
            logger.debug("State: Reachability heartbeat recorded (PONG)")

    def record_command_complete(self, success: bool) -> None:
        """
        Record that execution has finished.
        Explicit STOP terminates movement; motion state transitions truthfully to STOPPED.
        """
        with self._lock:
            self._execution_status = ExecutionState.IDLE if success else ExecutionState.ERROR
            if success:
                if self._current_command and self._current_command != "STOP":
                    self._last_command = self._current_command
                self._current_command = "STOP"
                self._motion_state = MotionState.STOPPED
                self._requested_command = None
                self._requested_speed = None
            logger.debug(f"State: Command completion recorded (success={success})")

    def record_telemetry(self, raw_payload: str | bytes) -> None:
        """
        Process a rich status packet or telemetry stream.
        """
        parsed = TelemetryParser.parse(raw_payload)
        if parsed.packet_type == TelemetryType.UNKNOWN:
            logger.warning(f"Rejected invalid telemetry: {parsed.error}")
            return

        with self._lock:
            now = datetime.now(timezone.utc)
            if parsed.connection:
                self._connection_status = parsed.connection
                self._last_heartbeat_timestamp = now

            if parsed.motion:
                self._motion_state = parsed.motion

            if parsed.command:
                self._current_command = parsed.command

            if parsed.speed is not None:
                self._requested_speed = parsed.speed

            self._last_telemetry_timestamp = now
            logger.debug(f"State: Telemetry recorded: {parsed}")

    def mark_disconnected(self) -> None:
        """
        Explicitly mark connection as DISCONNECTED.
        Decoupled: Watchdog status remains UNKNOWN (does not falsely claim EXPIRED).
        """
        with self._lock:
            self._connection_status = ConnectionStatus.DISCONNECTED
            self._last_heartbeat_timestamp = None
            self._execution_status = ExecutionState.IDLE
            logger.info("State: Marked connection as DISCONNECTED")

    async def probe_heartbeat(self, transport: Optional[object] = None) -> bool:
        """
        Send a PING to ESP32 on port 8888 using an independent ephemeral socket.
        Guarantees zero cross-consumption with Stage 6 execution packets.
        """
        from app.execution.transport import UDPTransport
        trans = transport or UDPTransport(timeout=1.0)
        try:
            _, resp_bytes, _ = await trans.send_and_receive("PING\n")
            if resp_bytes.strip() == b"PONG":
                self.record_heartbeat()
                return True
            else:
                logger.warning(f"Unexpected heartbeat response: {resp_bytes!r}")
                return False
        except Exception as e:
            logger.warning(f"Heartbeat probe failed: {e}")
            return False


# Canonical global singleton
default_state_manager = RobotStateManager()
