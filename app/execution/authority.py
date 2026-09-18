"""
AI Server Execution Authority Manager for IOFT Humanoid Robot.
Governs execution permissions for commands originating from the AI server.
Enforces confirmed STOP handshake on AI -> MANUAL transitions.
"""

import logging
from datetime import datetime, timezone
from typing import Tuple, Optional
from app.core.config import settings, ControlAuthority
from app.execution.errors import AuthorityConflictError
from app.execution.client import ESP32Client

logger = logging.getLogger("custom_llm_robot.execution.authority")


class ExecutionAuthorityManager:
    """
    Manages AI Server Execution Authority (MANUAL vs AI).
    IMPORTANT: Governs only commands originating from the AI server.
    Does NOT provide global hardware ownership over other external UDP clients.
    """

    def __init__(self, default_authority: Optional[ControlAuthority] = None):
        self._authority = default_authority or settings.ROBOT_DEFAULT_AUTHORITY
        self._changed_by = "system_init"
        self._changed_at = datetime.now(timezone.utc).isoformat()

    @property
    def current_authority(self) -> ControlAuthority:
        return self._authority

    @property
    def status_dict(self) -> dict:
        return {
            "current_authority": self._authority.value,
            "changed_by": self._changed_by,
            "changed_at": self._changed_at,
        }

    def is_execution_allowed(self, tool_name: str) -> Tuple[bool, str]:
        """
        Determine if command is authorized to execute physically.
        Emergency STOP is always permitted regardless of authority.
        """
        tool = (tool_name or "").lower().strip()
        if tool == "stop":
            return True, "Emergency STOP is permitted unconditionally in any authority state."

        if self._authority == ControlAuthority.AI:
            return True, "AI server execution authority is active."

        return (
            False,
            "AI Server Execution Authority is MANUAL. AI locomotion commands are blocked. "
            "Explicit transition to AI authority via POST /api/v1/commands/ownership is required."
        )

    async def set_authority(
        self,
        target_authority: ControlAuthority,
        changed_by: str = "api",
        esp32_client: Optional[ESP32Client] = None,
    ) -> Tuple[bool, str]:
        """
        Transition execution authority.
        On AI -> MANUAL: dispatches STOP|0|0.00 and only commits state if MOTOR_ACK is received.
        If STOP transmission or ACK fails, retains AI state and raises AuthorityConflictError.
        """
        if target_authority == self._authority:
            return True, f"Authority is already {self._authority.value}."

        # If switching from AI -> MANUAL, must confirm STOP first
        if self._authority == ControlAuthority.AI and target_authority == ControlAuthority.MANUAL:
            logger.info("Initiating AI -> MANUAL transition. Dispatching confirmed STOP handshake...")
            client = esp32_client or ESP32Client()
            try:
                hardware_affected, ack_received, msg = await client.emergency_stop()
                if not ack_received:
                    logger.error(f"AI -> MANUAL transition aborted: STOP was not acknowledged ({msg})")
                    raise AuthorityConflictError(
                        "Transition to MANUAL failed: STOP command was sent but not acknowledged by ESP32. "
                        "Retaining AI authority to prevent unverified physical state."
                    )
                logger.info("STOP acknowledged by ESP32. Committing authority transition to MANUAL.")
            except Exception as e:
                logger.error(f"AI -> MANUAL transition failed during STOP handshake: {e}")
                raise AuthorityConflictError(
                    f"Transition to MANUAL rejected: Failed to confirm STOP with ESP32: {e}"
                ) from e

        # Commit transition
        self._authority = target_authority
        self._changed_by = changed_by
        self._changed_at = datetime.now(timezone.utc).isoformat()
        logger.info(f"Execution authority transitioned to {self._authority.value} by '{changed_by}'.")
        return True, f"Authority successfully transitioned to {self._authority.value}."


# Singleton instance
default_authority_manager = ExecutionAuthorityManager()
