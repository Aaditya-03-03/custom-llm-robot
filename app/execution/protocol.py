"""
Wire protocol encoder and decoder for IOFT Humanoid Robot ESP32.
Implements the frozen hardware wire protocol: DIRECTION|SPEED|STEERING.
Zero changes to ESP32 firmware wire format.
"""

from typing import Dict, Any, Optional
from app.core.config import settings
from app.execution.errors import InvalidPacketError, CapabilityDisabledError

EXPECTED_ACK = "MOTOR_ACK"


class ESP32Protocol:
    """
    Translates validated robot commands into the exact wire protocol expected by ESP32 firmware.
    Wire format: DIRECTION|SPEED|STEERING
    """

    @classmethod
    def format_packet(cls, tool_name: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Encode tool name and parameters into the wire packet string.
        Enforces required speed for locomotion and pivot-turn physical verification gate.
        """
        tool = (tool_name or "").lower().strip()
        params = parameters or {}

        if tool == "stop":
            return "STOP|0|0.00"

        # Movement tools: forward, backward, left, right
        if tool in ("forward", "backward", "left", "right"):
            # 1. Pivot turn verification check
            if tool in ("left", "right") and not settings.ROBOT_PIVOT_TURNS_VERIFIED:
                raise CapabilityDisabledError(
                    f"Pivot turn capability ('{tool}') is unverified on physical hardware and disabled. "
                    "Set ROBOT_PIVOT_TURNS_VERIFIED=true after controlled physical testing."
                )

            # 2. Option A: Speed is strictly required for physical locomotion
            speed = params.get("speed")
            if speed is None:
                raise InvalidPacketError(
                    f"Physical locomotion tool '{tool}' requires an explicit 'speed' parameter (1-100)."
                )

            if not isinstance(speed, int) or isinstance(speed, bool) or not (1 <= speed <= 100):
                raise InvalidPacketError(
                    f"Invalid speed {speed!r}: speed must be an integer between 1 and 100."
                )

            direction = tool.upper()
            return f"{direction}|{speed}|0.00"

        raise InvalidPacketError(f"Tool '{tool_name}' cannot be translated to an ESP32 motor packet.")

    @classmethod
    def is_valid_ack(cls, data: bytes) -> bool:
        """Verify whether received UDP data matches expected MOTOR_ACK."""
        if not data:
            return False
        try:
            decoded = data.decode("utf-8", errors="ignore").strip()
            return decoded == EXPECTED_ACK
        except Exception:
            return False
