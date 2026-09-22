"""
Tests for Stage 9 Execution Lifecycle & Concurrency Invariants.
Verifies:
- Complete forward lifecycle transitions:
  CREATED -> VALIDATED -> AUTHORIZED -> DISPATCHED -> EXECUTING -> ACKNOWLEDGED -> COMPLETED -> VERIFYING -> VERIFIED / NOT_VERIFIED
- Failure lifecycle transitions:
  EXECUTING -> FAILED -> RECOVERY -> FAILED
- Invariant 3: EXECUTING phase denotes active transaction and does NOT imply physical motion
- Mandatory verification_deadline on creation
"""

import pytest
from datetime import datetime, timezone, timedelta

from app.core.config import settings
from app.schemas.commands import ExecutionStatus, PhysicalExecutionResponse
from app.state.models import RobotState, ConnectionStatus, MotionState, ExecutionState
from app.verification.models import (
    ExecutionPhase,
    VerificationStatus,
    RecoveryAction,
    ExecutionRecord,
)
from app.verification.manager import ExecutionManager


@pytest.mark.anyio
async def test_execution_record_creation_and_mandatory_deadline():
    """Verify ExecutionRecord initializes with CREATED phase and mandatory verification_deadline."""
    manager = ExecutionManager()
    started_before = datetime.now(timezone.utc)
    record = await manager.create_record(
        execution_id="exec_1",
        request_id="req_1",
        session_id="sess_1",
        plan_id="plan_1",
        action_id="act_1",
        command="FORWARD",
        speed=20,
        steps=1,
    )
    started_after = datetime.now(timezone.utc)

    assert record.execution_id == "exec_1"
    assert record.command == "FORWARD"
    assert record.speed == 20
    assert record.steps == 1
    assert record.execution_phase == ExecutionPhase.CREATED
    assert record.verification_status == VerificationStatus.PENDING
    assert record.physical_motion_verified is None

    # Mandatory deadline check
    assert record.started_at is not None
    assert record.verification_deadline is not None
    expected_deadline_min = started_before + timedelta(seconds=settings.ROBOT_VERIFICATION_TIMEOUT_SECONDS)
    expected_deadline_max = started_after + timedelta(seconds=settings.ROBOT_VERIFICATION_TIMEOUT_SECONDS)
    assert expected_deadline_min <= record.verification_deadline <= expected_deadline_max


@pytest.mark.anyio
async def test_forward_lifecycle_transitions_to_not_verified():
    """Verify normal locomotion transitions cleanly through all phases to terminal NOT_VERIFIED."""
    manager = ExecutionManager()
    record = await manager.create_record(
        execution_id="exec_seq",
        request_id="req_seq",
        session_id="sess_seq",
        plan_id="plan_seq",
        action_id="act_seq",
        command="FORWARD",
    )

    # 1. Validation & Authority & Dispatch
    await manager.advance_phase(record.execution_id, ExecutionPhase.VALIDATED)
    assert manager.get_record(record.execution_id).execution_phase == ExecutionPhase.VALIDATED

    await manager.advance_phase(record.execution_id, ExecutionPhase.AUTHORIZED)
    assert manager.get_record(record.execution_id).execution_phase == ExecutionPhase.AUTHORIZED

    await manager.advance_phase(record.execution_id, ExecutionPhase.DISPATCHED)
    assert manager.get_record(record.execution_id).execution_phase == ExecutionPhase.DISPATCHED

    # 2. Executing (active Stage 6 transaction)
    await manager.advance_phase(record.execution_id, ExecutionPhase.EXECUTING)
    r = manager.get_record(record.execution_id)
    assert r.execution_phase == ExecutionPhase.EXECUTING
    # Invariant 3: EXECUTING does NOT imply physical motion
    assert r.physical_motion_verified is None

    # 3. Stage 6 response & Acknowledged
    resp = PhysicalExecutionResponse(
        success=True,
        status=ExecutionStatus.EXECUTION_SUCCESS,
        execution_id=record.execution_id,
        tool="forward",
        parameters={"speed": 20},
        packet_sent="FORWARD|20|0.00",
        ack_received=True,
        hardware_state_affected=True,
        duration_seconds=0.5,
        message="Success",
    )
    await manager.record_stage6_result(record.execution_id, resp)
    await manager.advance_phase(record.execution_id, ExecutionPhase.ACKNOWLEDGED)
    assert manager.get_record(record.execution_id).ack_received is True
    assert manager.get_record(record.execution_id).execution_phase == ExecutionPhase.ACKNOWLEDGED

    # 4. Completed & Verifying
    await manager.advance_phase(record.execution_id, ExecutionPhase.COMPLETED)
    await manager.advance_phase(record.execution_id, ExecutionPhase.VERIFYING)
    assert manager.get_record(record.execution_id).execution_phase == ExecutionPhase.VERIFYING

    # 5. Terminal NOT_VERIFIED
    from app.verification.models import VerificationResult
    res = VerificationResult(
        verification_status=VerificationStatus.NOT_VERIFIED,
        physical_motion_verified=None,
        reason="Unverified",
    )
    await manager.complete_verification(record.execution_id, res, {"motion_state": "STOPPED"}, ExecutionPhase.NOT_VERIFIED)

    final_r = manager.get_record(record.execution_id)
    assert final_r.execution_phase == ExecutionPhase.NOT_VERIFIED
    assert final_r.verification_status == VerificationStatus.NOT_VERIFIED
    assert final_r.physical_motion_verified is None
    assert final_r.completed_at is not None


@pytest.mark.anyio
async def test_failure_lifecycle_transitions():
    """Verify failure transitions from EXECUTING -> RECOVERY -> FAILED."""
    manager = ExecutionManager()
    record = await manager.create_record(
        execution_id="exec_fail",
        request_id="req_fail",
        session_id="sess_fail",
        plan_id="plan_fail",
        action_id="act_fail",
        command="FORWARD",
    )
    await manager.advance_phase(record.execution_id, ExecutionPhase.EXECUTING)

    # Failure during execution
    await manager.advance_phase(record.execution_id, ExecutionPhase.RECOVERY)
    assert manager.get_record(record.execution_id).execution_phase == ExecutionPhase.RECOVERY

    await manager.record_recovery(record.execution_id, RecoveryAction.STOP, "Emergency STOP dispatched")

    from app.verification.models import VerificationResult
    res = VerificationResult(
        verification_status=VerificationStatus.FAILED,
        physical_motion_verified=None,
        reason="ACK timeout",
    )
    await manager.complete_verification(record.execution_id, res, {"motion_state": "STOPPED"}, ExecutionPhase.FAILED)

    final_r = manager.get_record(record.execution_id)
    assert final_r.execution_phase == ExecutionPhase.FAILED
    assert final_r.verification_status == VerificationStatus.FAILED
    assert final_r.recovery_action == RecoveryAction.STOP
    assert "Emergency STOP" in final_r.recovery_result
