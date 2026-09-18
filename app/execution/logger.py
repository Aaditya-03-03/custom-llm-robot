"""
Structured audit logger for robot physical execution records.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from app.schemas.commands import ExecutionStatus

logger = logging.getLogger("custom_llm_robot.execution.audit")


class ExecutionLogger:
    """
    Logs structured physical execution records for hardware telemetry and debugging.
    """

    @classmethod
    def log_execution(
        cls,
        execution_id: str,
        tool: str,
        parameters: Dict[str, Any],
        status: ExecutionStatus,
        hardware_state_affected: bool,
        ack_received: bool,
        packet_sent: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        message: str = "",
        errors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_id": execution_id,
            "source": "ai_server",
            "tool": tool,
            "parameters": parameters,
            "packet_sent": packet_sent,
            "hardware_state_affected": hardware_state_affected,
            "ack_received": ack_received,
            "status": status.value,
            "duration_seconds": duration_seconds,
            "message": message,
            "errors": errors or [],
        }

        if status == ExecutionStatus.EXECUTION_SUCCESS:
            logger.info(f"AUDIT EXECUTION [SUCCESS]: {record}")
        elif status in (ExecutionStatus.ACK_TIMEOUT, ExecutionStatus.EXECUTION_BUSY):
            logger.warning(f"AUDIT EXECUTION [{status.value}]: {record}")
        else:
            logger.error(f"AUDIT EXECUTION [{status.value}]: {record}")

        return record
