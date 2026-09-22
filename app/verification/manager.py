"""
Execution Record Lifecycle Manager for Stage 9.
Maintains in-memory thread-safe storage of ExecutionRecords,
coordinates lifecycle phase transitions, and provides read-only lookups.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List, Any

from app.core.config import settings
from app.schemas.commands import ExecutionStatus, PhysicalExecutionResponse
from app.verification.models import (
    ExecutionPhase,
    ExecutionRecord,
    RecoveryAction,
    VerificationResult,
    VerificationStatus,
)
from app.verification.verifier import get_expected_robot_state

logger = logging.getLogger("custom_llm_robot.verification.manager")


class ExecutionManager:
    """
    Coordinates lifecycle state transitions and audit storage for closed-loop execution.
    """

    def __init__(self):
        self._records: Dict[str, ExecutionRecord] = {}
        self._lock = asyncio.Lock()

    async def create_record(
        self,
        execution_id: str,
        request_id: str,
        session_id: str,
        plan_id: str,
        action_id: str,
        command: str,
        speed: Optional[int] = None,
        steps: Optional[int] = None,
    ) -> ExecutionRecord:
        """
        Initialize and register a new ExecutionRecord with a mandatory bounded observation deadline.
        """
        started_at = datetime.now(timezone.utc)
        deadline = started_at + timedelta(seconds=settings.ROBOT_VERIFICATION_TIMEOUT_SECONDS)
        expected_state = get_expected_robot_state(command, ExecutionStatus.EXECUTION_SUCCESS)

        record = ExecutionRecord(
            execution_id=execution_id,
            request_id=request_id,
            session_id=session_id,
            plan_id=plan_id,
            action_id=action_id,
            command=command,
            speed=speed,
            steps=steps,
            execution_phase=ExecutionPhase.CREATED,
            execution_status=ExecutionStatus.EXECUTION_SUCCESS,
            verification_status=VerificationStatus.PENDING,
            physical_motion_verified=None,
            expected_execution_status=ExecutionStatus.EXECUTION_SUCCESS,
            expected_robot_state=expected_state,
            observed_state=None,
            ack_received=False,
            recovery_action=None,
            recovery_result=None,
            started_at=started_at,
            verification_deadline=deadline,
            completed_at=None,
        )

        async with self._lock:
            self._records[execution_id] = record

        logger.debug(f"Created ExecutionRecord [{execution_id}] for action '{action_id}' ({command})")
        return record

    async def advance_phase(
        self,
        execution_id: str,
        phase: ExecutionPhase,
    ) -> Optional[ExecutionRecord]:
        """
        Advance execution lifecycle phase.
        """
        async with self._lock:
            record = self._records.get(execution_id)
            if not record:
                return None
            record.execution_phase = phase
            return record

    async def record_stage6_result(
        self,
        execution_id: str,
        resp: PhysicalExecutionResponse,
    ) -> Optional[ExecutionRecord]:
        """
        Record authoritative Stage 6 execution outcome and ACK status.
        """
        async with self._lock:
            record = self._records.get(execution_id)
            if not record:
                return None
            record.execution_status = resp.status
            record.ack_received = resp.ack_received
            return record

    async def complete_verification(
        self,
        execution_id: str,
        result: VerificationResult,
        observed_state: Optional[Dict[str, Any]],
        final_phase: ExecutionPhase,
    ) -> Optional[ExecutionRecord]:
        """
        Finalize execution record with verification result and terminal phase.
        """
        async with self._lock:
            record = self._records.get(execution_id)
            if not record:
                return None
            record.verification_status = result.verification_status
            record.physical_motion_verified = result.physical_motion_verified
            record.observed_state = observed_state
            record.execution_phase = final_phase
            record.completed_at = datetime.now(timezone.utc)
            return record

    async def record_recovery(
        self,
        execution_id: str,
        action: RecoveryAction,
        outcome: str,
    ) -> Optional[ExecutionRecord]:
        """
        Record controlled recovery action taken.
        """
        async with self._lock:
            record = self._records.get(execution_id)
            if not record:
                return None
            record.recovery_action = action
            record.recovery_result = outcome
            return record

    def get_record(self, execution_id: str) -> Optional[ExecutionRecord]:
        """
        Read-only retrieval of execution record.
        Strictly in-memory; sends zero UDP packets.
        """
        return self._records.get(execution_id)

    def list_records(self) -> List[ExecutionRecord]:
        """
        Read-only list of all tracked execution records.
        """
        return list(self._records.values())

    def clear(self) -> None:
        """
        Clear in-memory records (useful for test isolation).
        """
        self._records.clear()


default_execution_manager = ExecutionManager()
