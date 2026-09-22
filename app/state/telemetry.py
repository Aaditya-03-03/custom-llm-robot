"""
Telemetry Parser for IOFT Humanoid Robot.
Decodes raw ESP32 packet responses: MOTOR_ACK, PONG, pipe-delimited STATUS, and JSON telemetry.
Enforces fail-closed validation on malformed telemetry packets.
"""

import json
import logging
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel

from app.state.models import ConnectionStatus, MotionState

logger = logging.getLogger("custom_llm_robot.state.telemetry")


class TelemetryType(str, Enum):
    MOTOR_ACK = "MOTOR_ACK"
    PONG = "PONG"
    STATUS = "STATUS"
    UNKNOWN = "UNKNOWN"


class ParsedTelemetry(BaseModel):
    packet_type: TelemetryType
    raw_payload: str
    connection: Optional[ConnectionStatus] = None
    motion: Optional[MotionState] = None
    command: Optional[str] = None
    speed: Optional[int] = None
    error: Optional[str] = None


class TelemetryParser:
    """
    Decodes incoming raw byte buffers or strings from the ESP32.
    """

    @staticmethod
    def parse(payload: str | bytes) -> ParsedTelemetry:
        if isinstance(payload, bytes):
            text = payload.decode("utf-8", errors="ignore").strip()
        else:
            text = (payload or "").strip()

        if not text:
            return ParsedTelemetry(
                packet_type=TelemetryType.UNKNOWN,
                raw_payload="",
                error="Empty telemetry payload",
            )

        # 1. Standard In-band ACKs
        if text == "MOTOR_ACK":
            return ParsedTelemetry(
                packet_type=TelemetryType.MOTOR_ACK,
                raw_payload=text,
                connection=ConnectionStatus.CONNECTED,
            )

        # 2. Reachability Heartbeat
        if text == "PONG":
            return ParsedTelemetry(
                packet_type=TelemetryType.PONG,
                raw_payload=text,
                connection=ConnectionStatus.CONNECTED,
            )

        # 3. Pipe-delimited Status: STATUS|<connection>|<motion>|<command>|<speed>
        if text.startswith("STATUS|"):
            parts = text.split("|")
            if len(parts) < 5:
                logger.warning(f"Malformed STATUS packet (too few fields): '{text}'")
                return ParsedTelemetry(
                    packet_type=TelemetryType.UNKNOWN,
                    raw_payload=text,
                    error="Malformed pipe-delimited STATUS packet: expected 5 fields",
                )

            try:
                conn_str = parts[1].upper()
                motion_str = parts[2].upper()
                cmd_str = parts[3].upper()
                speed_int = int(parts[4])

                conn = ConnectionStatus(conn_str) if conn_str in ConnectionStatus._value2member_map_ else ConnectionStatus.UNKNOWN
                motion = MotionState(motion_str) if motion_str in MotionState._value2member_map_ else MotionState.UNKNOWN

                return ParsedTelemetry(
                    packet_type=TelemetryType.STATUS,
                    raw_payload=text,
                    connection=conn,
                    motion=motion,
                    command=cmd_str,
                    speed=speed_int,
                )
            except Exception as e:
                logger.warning(f"Failed to parse pipe-delimited STATUS packet '{text}': {e}")
                return ParsedTelemetry(
                    packet_type=TelemetryType.UNKNOWN,
                    raw_payload=text,
                    error=f"Error parsing fields: {e}",
                )

        # 4. JSON Status Telemetry
        if text.startswith("{") and text.endswith("}"):
            try:
                data = json.loads(text)
                if not isinstance(data, dict):
                    raise ValueError("JSON must be an object")

                packet_type_val = data.get("type", "").upper()
                if packet_type_val == "STATUS":
                    conn_val = data.get("connection", "").upper()
                    motion_val = data.get("motion", "").upper()
                    cmd_val = data.get("command", "").upper() if data.get("command") else None
                    speed_val = data.get("speed")

                    conn = ConnectionStatus(conn_val) if conn_val in ConnectionStatus._value2member_map_ else None
                    motion = MotionState(motion_val) if motion_val in MotionState._value2member_map_ else None

                    return ParsedTelemetry(
                        packet_type=TelemetryType.STATUS,
                        raw_payload=text,
                        connection=conn,
                        motion=motion,
                        command=cmd_val,
                        speed=int(speed_val) if speed_val is not None else None,
                    )
            except Exception as e:
                logger.warning(f"Failed to parse JSON telemetry '{text}': {e}")
                return ParsedTelemetry(
                    packet_type=TelemetryType.UNKNOWN,
                    raw_payload=text,
                    error=f"Malformed JSON telemetry: {e}",
                )

        # 5. Unrecognized payload
        logger.debug(f"Unrecognized telemetry packet: '{text}'")
        return ParsedTelemetry(
            packet_type=TelemetryType.UNKNOWN,
            raw_payload=text,
            error=f"Unrecognized telemetry packet format: '{text}'",
        )
