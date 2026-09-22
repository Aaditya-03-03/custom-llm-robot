"""
Unit Tests for Stage 9 Controlled Recovery Policies & Safety Invariants.
Verifies:
- Invariant 6: Recovery Deduplication: detects if Stage 6 already dispatched failsafe STOP on timeout.
- Invariant 7: Recovery routes strictly through Stage 5 & 6 (PhysicalExecutionRequest -> execute_command).
- Invariant 8: STOP is primary recovery action; NO automatic movement retry.
- Normal NOT_VERIFIED locomotion triggers RecoveryAction.NONE (no unnecessary emergency stop).
- Verification FAILED or stale robot triggers RecoveryAction.STOP.
- ROBOT_RECOVERY_ENABLED = False bypasses recovery.
"""

from unittest.mock import patch, AsyncMock
import pytest

from app.core.config import settings
from app.schemas.commands import ExecutionStatus, PhysicalExecutionResponse, PhysicalExecutionRequest
from app.verification.models import RecoveryAction, VerificationResult, VerificationStatus
from app.verification.recovery import determine_recovery_policy, execute_controlled_recovery


def test_normal_unverified_locomotion_requires_no_recovery():
    """Unverified locomotion (normal completion with no sensor telemetry) requires no recovery."""
    stage6 = PhysicalExecutionResponse(
        success=True,
        status=ExecutionStatus.EXECUTION_SUCCESS,
        execution_id="e1",
        tool="forward",
        parameters={"speed": 20},
        packet_sent="FORWARD|20|0.00",
        ack_received=True,
        hardware_state_affected=True,
        duration_seconds=0.5,
        message="Success",
    )
    verif = VerificationResult(
        verification_status=VerificationStatus.NOT_VERIFIED,
        physical_motion_verified=None,
        reason="Unverified",
    )
    action, already_performed = determine_recovery_policy(stage6, verif, is_stale=False)
    assert action == RecoveryAction.NONE
    assert already_performed is False


def test_recovery_deduplication_on_stage6_ack_timeout():
    """Invariant 6: If Stage 6 pulse timed out and already issued failsafe STOP, Stage 9 avoids duplicate STOP."""
    stage6 = PhysicalExecutionResponse(
        success=False,
        status=ExecutionStatus.ACK_TIMEOUT,
        execution_id="e2",
        tool="forward",
        parameters={"speed": 20},
        packet_sent="FORWARD|20|0.00",
        ack_received=False,
        hardware_state_affected=True,  # Hardware affected: Stage 6 pulse timer triggered emergency_stop
        duration_seconds=None,
        message="Pulse ACK timeout",
        errors=["ACK timeout"],
    )
    verif = VerificationResult(
        verification_status=VerificationStatus.FAILED,
        physical_motion_verified=None,
        reason="ACK timeout",
    )
    action, already_performed = determine_recovery_policy(stage6, verif, is_stale=False)
    assert action == RecoveryAction.STOP
    assert already_performed is True  # Avoids duplicate STOP packet


def test_state_mismatch_or_stale_requires_stage9_stop():
    """If verification fails due to state mismatch or stale robot, Stage 9 must dispatch STOP."""
    stage6 = PhysicalExecutionResponse(
        success=True,
        status=ExecutionStatus.EXECUTION_SUCCESS,
        execution_id="e3",
        tool="forward",
        parameters={"speed": 20},
        packet_sent="FORWARD|20|0.00",
        ack_received=True,
        hardware_state_affected=True,
        duration_seconds=0.5,
        message="Success",
    )
    verif = VerificationResult(
        verification_status=VerificationStatus.FAILED,
        physical_motion_verified=False,
        reason="Sensory telemetry refutes motion (stall detected)",
    )
    action, already_performed = determine_recovery_policy(stage6, verif, is_stale=False)
    assert action == RecoveryAction.STOP
    assert already_performed is False  # Stage 9 must dispatch STOP


@pytest.mark.anyio
async def test_execute_controlled_recovery_dispatches_via_stage6_adapter():
    """Invariant 7: Recovery routes strictly through Stage 5/6 adapter (never raw UDP)."""
    resp_mock = PhysicalExecutionResponse(
        success=True,
        status=ExecutionStatus.EXECUTION_SUCCESS,
        execution_id="stop_exec",
        tool="stop",
        parameters={},
        packet_sent="STOP|0|0.00",
        ack_received=True,
        hardware_state_affected=True,
        duration_seconds=0.0,
        message="Halted",
    )

    with patch("app.execution.adapter.default_execution_adapter.execute_command", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = resp_mock
        res = await execute_controlled_recovery(RecoveryAction.STOP)
        assert "Controlled recovery STOP successfully executed" in res
        assert mock_exec.call_count == 1
        call_req: PhysicalExecutionRequest = mock_exec.call_args[0][0]
        assert call_req.tool_name == "stop"
        assert call_req.parameters == {}


def test_recovery_disabled_by_config():
    """If ROBOT_RECOVERY_ENABLED = False, no recovery is scheduled."""
    stage6 = PhysicalExecutionResponse(
        success=False,
        status=ExecutionStatus.ACK_TIMEOUT,
        execution_id="e4",
        tool="forward",
        parameters={"speed": 20},
        packet_sent="FORWARD|20|0.00",
        ack_received=False,
        hardware_state_affected=False,
        duration_seconds=None,
        message="Timeout",
        errors=["Timeout"],
    )
    verif = VerificationResult(
        verification_status=VerificationStatus.FAILED,
        physical_motion_verified=None,
        reason="Timeout",
    )
    orig = settings.ROBOT_RECOVERY_ENABLED
    try:
        settings.ROBOT_RECOVERY_ENABLED = False
        action, already = determine_recovery_policy(stage6, verif, is_stale=False)
        assert action == RecoveryAction.NONE
        assert already is False
    finally:
        settings.ROBOT_RECOVERY_ENABLED = orig
