"""
Unit Tests for Stage 9 Deterministic Verification Rules & Telemetry Correlation.
Verifies:
- Invariant 1 & 5: MOTOR_ACK produces NOT_VERIFIED and physical_motion_verified = None.
- Invariant 4: STOP command produces controller-level VERIFIED and physical_motion_verified = None.
- Sensory Confirmation: Valid IMU/encoder telemetry within window produces VERIFIED (physical_motion_verified = True).
- Sensory Refutation: Zero motion or stall telemetry produces FAILED (physical_motion_verified = False).
- Invariant 2 (Bounded Window Boundary Tests):
  * timestamp == deadline -> accepted
  * timestamp == deadline + 1ms -> rejected
  * timestamp < started_at -> rejected
  * Old Execution A telemetry cannot verify Execution B
- Invariant 9: ROBOT_VERIFICATION_ENABLED = False returns NOT_VERIFIED gracefully.
- State mismatch & disconnection produces FAILED.
"""

import pytest
from datetime import datetime, timezone, timedelta

from app.core.config import settings
from app.schemas.commands import ExecutionStatus, PhysicalExecutionResponse
from app.state.models import RobotState, ConnectionStatus, MotionState, ExecutionState
from app.verification.models import ExecutionRecord, VerificationStatus, ExecutionPhase
from app.verification.verifier import verify_execution, get_expected_robot_state


def _make_record(command: str = "FORWARD", started_at: datetime = None, duration_sec: float = 2.0) -> ExecutionRecord:
    start = started_at or datetime.now(timezone.utc)
    deadline = start + timedelta(seconds=duration_sec)
    return ExecutionRecord(
        execution_id="test_exec",
        request_id="req_1",
        session_id="sess_1",
        plan_id="plan_1",
        action_id="act_1",
        command=command,
        started_at=start,
        verification_deadline=deadline,
        execution_status=ExecutionStatus.EXECUTION_SUCCESS,
        expected_execution_status=ExecutionStatus.EXECUTION_SUCCESS,
        expected_robot_state=get_expected_robot_state(command, ExecutionStatus.EXECUTION_SUCCESS),
    )


def _make_stage6_resp(success: bool = True, status: ExecutionStatus = ExecutionStatus.EXECUTION_SUCCESS) -> PhysicalExecutionResponse:
    return PhysicalExecutionResponse(
        success=success,
        status=status,
        execution_id="test_exec",
        tool="forward",
        parameters={"speed": 20},
        packet_sent="FORWARD|20|0.00",
        ack_received=success,
        hardware_state_affected=True,
        duration_seconds=0.5,
        message="Executed",
    )


def test_motor_ack_locomotion_produces_not_verified():
    """Invariant 1 & 5: MOTOR_ACK confirms command processing, but physical movement remains UNKNOWN."""
    rec = _make_record(command="FORWARD")
    stage6 = _make_stage6_resp(success=True)
    before = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.STOPPED)
    after = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.STOPPED)

    res = verify_execution(rec, stage6, before, after, telemetry_samples=[])
    assert res.verification_status == VerificationStatus.NOT_VERIFIED
    assert res.physical_motion_verified is None
    assert "no sensor telemetry provided" in res.reason


def test_stop_command_produces_controller_level_verified():
    """Invariant 4: STOP + STOP ACK + STOPPED produces VERIFIED with physical_motion_verified = None."""
    rec = _make_record(command="STOP")
    stage6 = _make_stage6_resp(success=True)
    before = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.UNKNOWN)
    after = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.STOPPED)

    res = verify_execution(rec, stage6, before, after, telemetry_samples=[])
    assert res.verification_status == VerificationStatus.VERIFIED
    assert res.physical_motion_verified is None
    assert "Controller-level stop verified" in res.reason


def test_stop_command_with_explicit_sensory_standstill():
    """STOP with sensor measurement (standstill=True) elevates physical_motion_verified to True."""
    rec = _make_record(command="STOP")
    stage6 = _make_stage6_resp(success=True)
    before = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.UNKNOWN)
    after = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.STOPPED)

    sensor_telemetry = [{"timestamp": rec.started_at + timedelta(milliseconds=100), "standstill": True}]
    res = verify_execution(rec, stage6, before, after, telemetry_samples=sensor_telemetry)
    assert res.verification_status == VerificationStatus.VERIFIED
    assert res.physical_motion_verified is True
    assert "Physical stop confirmed" in res.reason


def test_locomotion_with_affirmative_imu_telemetry_verifies():
    """Locomotion with IMU delta inside observation window elevates to VERIFIED with True."""
    rec = _make_record(command="FORWARD")
    stage6 = _make_stage6_resp(success=True)
    before = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.STOPPED)
    after = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.STOPPED)

    sensor_telemetry = [{"timestamp": rec.started_at + timedelta(milliseconds=200), "imu_delta": 1.25}]
    res = verify_execution(rec, stage6, before, after, telemetry_samples=sensor_telemetry)
    assert res.verification_status == VerificationStatus.VERIFIED
    assert res.physical_motion_verified is True
    assert "Physical locomotion confirmed" in res.reason


def test_locomotion_with_refuted_motion_fails():
    """Locomotion where telemetry explicitly indicates zero motion or stall produces FAILED with False."""
    rec = _make_record(command="FORWARD")
    stage6 = _make_stage6_resp(success=True)
    before = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.STOPPED)
    after = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.STOPPED)

    sensor_telemetry = [{"timestamp": rec.started_at + timedelta(milliseconds=200), "stall_detected": True}]
    res = verify_execution(rec, stage6, before, after, telemetry_samples=sensor_telemetry)
    assert res.verification_status == VerificationStatus.FAILED
    assert res.physical_motion_verified is False
    assert "refutes physical locomotion" in res.reason


def test_observation_window_exact_deadline_boundary_accepted():
    """Sample exactly at verification_deadline (<=) is accepted."""
    rec = _make_record(command="FORWARD")
    stage6 = _make_stage6_resp(success=True)
    before = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.STOPPED)
    after = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.STOPPED)

    # Exactly at deadline
    sensor_telemetry = [{"timestamp": rec.verification_deadline, "motion_detected": True}]
    res = verify_execution(rec, stage6, before, after, telemetry_samples=sensor_telemetry)
    assert res.verification_status == VerificationStatus.VERIFIED
    assert res.physical_motion_verified is True


def test_observation_window_post_deadline_boundary_rejected():
    """Sample 1 millisecond after verification_deadline is rejected and discarded."""
    rec = _make_record(command="FORWARD")
    stage6 = _make_stage6_resp(success=True)
    before = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.STOPPED)
    after = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.STOPPED)

    # 1 ms past deadline
    past_sample = rec.verification_deadline + timedelta(milliseconds=1)
    sensor_telemetry = [{"timestamp": past_sample, "motion_detected": True}]
    res = verify_execution(rec, stage6, before, after, telemetry_samples=sensor_telemetry)
    # The out-of-window sample is discarded, so locomotion remains NOT_VERIFIED
    assert res.verification_status == VerificationStatus.NOT_VERIFIED
    assert res.physical_motion_verified is None


def test_observation_window_pre_started_at_boundary_rejected():
    """Sample before started_at is rejected and discarded."""
    rec = _make_record(command="FORWARD")
    stage6 = _make_stage6_resp(success=True)
    before = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.STOPPED)
    after = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.STOPPED)

    # 1 ms before started_at
    prior_sample = rec.started_at - timedelta(milliseconds=1)
    sensor_telemetry = [{"timestamp": prior_sample, "motion_detected": True}]
    res = verify_execution(rec, stage6, before, after, telemetry_samples=sensor_telemetry)
    assert res.verification_status == VerificationStatus.NOT_VERIFIED
    assert res.physical_motion_verified is None


def test_old_telemetry_from_execution_a_does_not_verify_execution_b():
    """Old telemetry recorded during Execution A must not verify Execution B."""
    # Execution A
    time_a_start = datetime(2026, 9, 22, 10, 0, 0, tzinfo=timezone.utc)
    rec_a = _make_record(command="FORWARD", started_at=time_a_start)
    telemetry_a = [{"timestamp": time_a_start + timedelta(seconds=1), "motion_detected": True}]

    # Execution B starts later
    time_b_start = datetime(2026, 9, 22, 10, 5, 0, tzinfo=timezone.utc)
    rec_b = _make_record(command="FORWARD", started_at=time_b_start)

    stage6 = _make_stage6_resp(success=True)
    before = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.STOPPED)
    after = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.STOPPED)

    # Attempt to verify Execution B using Execution A's telemetry
    res_b = verify_execution(rec_b, stage6, before, after, telemetry_samples=telemetry_a)
    assert res_b.verification_status == VerificationStatus.NOT_VERIFIED
    assert res_b.physical_motion_verified is None


def test_disabled_verification_fallback():
    """Invariant 9: ROBOT_VERIFICATION_ENABLED = False returns NOT_VERIFIED without failure."""
    rec = _make_record(command="FORWARD")
    stage6 = _make_stage6_resp(success=True)
    before = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.STOPPED)
    after = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.STOPPED)

    original_val = settings.ROBOT_VERIFICATION_ENABLED
    try:
        settings.ROBOT_VERIFICATION_ENABLED = False
        res = verify_execution(rec, stage6, before, after, telemetry_samples=[])
        assert res.verification_status == VerificationStatus.NOT_VERIFIED
        assert res.physical_motion_verified is None
        assert "disabled by configuration" in res.reason
    finally:
        settings.ROBOT_VERIFICATION_ENABLED = original_val


def test_stale_or_disconnected_robot_fails_verification():
    """Disconnection or stale state produces VerificationStatus.FAILED."""
    rec = _make_record(command="FORWARD")
    stage6 = _make_stage6_resp(success=True)
    before = RobotState(connection_status=ConnectionStatus.CONNECTED, motion_state=MotionState.STOPPED)
    after = RobotState(connection_status=ConnectionStatus.DISCONNECTED, motion_state=MotionState.STOPPED)

    res = verify_execution(rec, stage6, before, after, telemetry_samples=[])
    assert res.verification_status == VerificationStatus.FAILED
    assert "disconnected or state became stale" in res.reason
