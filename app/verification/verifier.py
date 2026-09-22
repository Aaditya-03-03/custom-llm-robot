"""
Pure, deterministic verification engine for Stage 9.
Evaluates expected vs observed robot states and correlates physical telemetry
strictly within the bounded execution observation window.
Enforces Invariants 1-5:
- MOTOR_ACK != Physical movement confirmed
- Bounded observation window freshness (started_at <= timestamp <= verification_deadline)
- Controller-level STOP verification != physical stop verification
- Evidence-based physical motion (True | False | None)
- Unknown remains UNKNOWN
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from app.core.config import settings
from app.schemas.commands import ExecutionStatus, PhysicalExecutionResponse
from app.state.models import RobotState, ConnectionStatus, MotionState, ExecutionState
from app.verification.models import (
    ExecutionRecord,
    VerificationResult,
    VerificationStatus,
)

logger = logging.getLogger("custom_llm_robot.verification.verifier")


def get_expected_robot_state(command: str, execution_status: ExecutionStatus) -> Dict[str, Any]:
    """
    Deterministic mapping from command and execution status to expected robot state.
    Note: Expected STOPPED state is based on Stage 6's explicit completion STOP,
    not on MOTOR_ACK from the original locomotion command.
    """
    cmd = (command or "").strip().upper()
    if execution_status != ExecutionStatus.EXECUTION_SUCCESS:
        return {
            "execution_status": ExecutionState.ERROR.value,
            "motion_state": MotionState.UNKNOWN.value,
        }

    # Successful locomotion halts after timed step duration with an explicit STOP
    if cmd in ("FORWARD", "BACKWARD", "LEFT", "RIGHT"):
        return {
            "execution_status": ExecutionState.IDLE.value,
            "motion_state": MotionState.STOPPED.value,
        }

    # Successful STOP explicitly halts and settles to IDLE
    if cmd == "STOP":
        return {
            "execution_status": ExecutionState.IDLE.value,
            "motion_state": MotionState.STOPPED.value,
        }

    # Default fallback for status/other commands
    return {
        "execution_status": ExecutionState.IDLE.value,
        "motion_state": MotionState.STOPPED.value,
    }


def verify_execution(
    execution_record: ExecutionRecord,
    stage6_resp: PhysicalExecutionResponse,
    state_before: RobotState,
    state_after: RobotState,
    telemetry_samples: Optional[List[Dict[str, Any]]] = None,
) -> VerificationResult:
    """
    Deterministic verification engine:
    Expected + Observed + Telemetry -> VerificationResult.
    """
    cmd = (execution_record.command or "").strip().upper()
    expected_exec_status = execution_record.expected_execution_status
    expected_state = get_expected_robot_state(cmd, expected_exec_status)
    observed_state_dict = state_after.model_dump()

    # Invariant 9: If verification is globally disabled, return NOT_VERIFIED gracefully without failure
    if not settings.ROBOT_VERIFICATION_ENABLED:
        logger.info(f"Verification disabled by config for execution [{execution_record.execution_id}]")
        return VerificationResult(
            verification_status=VerificationStatus.NOT_VERIFIED,
            physical_motion_verified=None,
            reason="Verification disabled by configuration.",
            expected_execution_status=expected_exec_status,
            expected_robot_state=expected_state,
            observed_robot_state=observed_state_dict,
        )

    # 1. Filter telemetry strictly within the bounded observation window
    # Invariant 2: execution_record.started_at <= sample.timestamp <= execution_record.verification_deadline
    valid_telemetry: List[Dict[str, Any]] = []
    if telemetry_samples:
        for sample in telemetry_samples:
            ts = sample.get("timestamp")
            if isinstance(ts, datetime):
                if execution_record.started_at <= ts <= execution_record.verification_deadline:
                    valid_telemetry.append(sample)
                else:
                    logger.debug(
                        f"Discarded out-of-window telemetry sample (ts={ts}, "
                        f"window=[{execution_record.started_at}, {execution_record.verification_deadline}])"
                    )

    # 2. Check Stage 6 Execution Status
    if stage6_resp.status != ExecutionStatus.EXECUTION_SUCCESS or not stage6_resp.success:
        err_msg = f"Stage 6 execution failed with status {stage6_resp.status}: {stage6_resp.message}"
        logger.warning(f"Verification FAILED for [{execution_record.execution_id}]: {err_msg}")
        return VerificationResult(
            verification_status=VerificationStatus.FAILED,
            physical_motion_verified=None,
            reason=err_msg,
            expected_execution_status=expected_exec_status,
            expected_robot_state=expected_state,
            observed_robot_state=observed_state_dict,
        )

    # 3. Check State Connectivity & Freshness
    # Locomotion requires fresh connection; emergency STOP failsafe is allowed to execute and verify even when stale
    if cmd != "STOP" and (state_after.connection_status == ConnectionStatus.DISCONNECTED or state_after.is_stale):
        err_msg = "Robot connection is disconnected or state became stale during execution."
        logger.warning(f"Verification FAILED for [{execution_record.execution_id}]: {err_msg}")
        return VerificationResult(
            verification_status=VerificationStatus.FAILED,
            physical_motion_verified=None,
            reason=err_msg,
            expected_execution_status=expected_exec_status,
            expected_robot_state=expected_state,
            observed_robot_state=observed_state_dict,
        )

    # 4. Check Observed State for Unexpected Errors (STOP is the failsafe to halt any error state)
    if cmd != "STOP" and state_after.execution_status == ExecutionState.ERROR:
        err_msg = "Observed robot reported ERROR execution state."
        logger.warning(f"Verification FAILED for [{execution_record.execution_id}]: {err_msg}")
        return VerificationResult(
            verification_status=VerificationStatus.FAILED,
            physical_motion_verified=None,
            reason=err_msg,
            expected_execution_status=expected_exec_status,
            expected_robot_state=expected_state,
            observed_robot_state=observed_state_dict,
        )

    # 5. Evaluate STOP Command
    # Invariant 4: Controller-level STOP verification != physical stop verification
    if cmd == "STOP":
        if state_after.motion_state == MotionState.STOPPED:
            # Check if explicit physical sensor telemetry exists confirming standstill
            physical_verified: Optional[bool] = None
            reason = "Controller-level stop verified; physical motion remains UNKNOWN without sensor confirmation."
            for s in valid_telemetry:
                if s.get("standstill") is True or s.get("velocity") == 0.0:
                    physical_verified = True
                    reason = "Physical stop confirmed by sensory telemetry."
                    break

            return VerificationResult(
                verification_status=VerificationStatus.VERIFIED,
                physical_motion_verified=physical_verified,
                reason=reason,
                expected_execution_status=expected_exec_status,
                expected_robot_state=expected_state,
                observed_robot_state=observed_state_dict,
            )
        elif state_after.motion_state == MotionState.UNKNOWN:
            # Table: Expected STOPPED, Observed UNKNOWN -> Result: Unknown / NOT_VERIFIED
            return VerificationResult(
                verification_status=VerificationStatus.NOT_VERIFIED,
                physical_motion_verified=None,
                reason="STOP command executed at controller level, but observed motion state remains UNKNOWN.",
                expected_execution_status=expected_exec_status,
                expected_robot_state=expected_state,
                observed_robot_state=observed_state_dict,
            )
        else:
            return VerificationResult(
                verification_status=VerificationStatus.FAILED,
                physical_motion_verified=None,
                reason=f"STOP command completed but motion_state is {state_after.motion_state}.",
                expected_execution_status=expected_exec_status,
                expected_robot_state=expected_state,
                observed_robot_state=observed_state_dict,
            )

    # 6. Evaluate Locomotion Commands (FORWARD, BACKWARD, LEFT, RIGHT)
    # The command completed its timed steps and Stage 6 issued completion STOP
    if state_after.motion_state in (MotionState.STOPPED, MotionState.UNKNOWN):
        # Check sensory telemetry in observation window
        physical_verified = None
        motion_refuted = False
        motion_confirmed = False

        for s in valid_telemetry:
            # Check for affirmative physical motion
            if (
                s.get("motion_detected") is True
                or s.get("motion_state") == "MOVING"
                or (isinstance(s.get("imu_delta"), (int, float)) and s["imu_delta"] > 0)
                or (isinstance(s.get("encoder_delta"), (int, float)) and s["encoder_delta"] > 0)
            ):
                motion_confirmed = True

            # Check for refuted physical motion (e.g. zero motion or stall detected during commanded run)
            if s.get("motion_detected") is False or s.get("stall_detected") is True:
                motion_refuted = True

        if motion_refuted:
            # Physical evidence indicates motion did not occur
            logger.warning(f"Verification FAILED for [{execution_record.execution_id}]: sensory telemetry refutes motion")
            return VerificationResult(
                verification_status=VerificationStatus.FAILED,
                physical_motion_verified=False,
                reason="Sensory telemetry refutes physical locomotion (zero motion / stall detected).",
                expected_execution_status=expected_exec_status,
                expected_robot_state=expected_state,
                observed_robot_state=observed_state_dict,
            )
        elif motion_confirmed:
            # Physical evidence confirms motion
            logger.info(f"Verification VERIFIED for [{execution_record.execution_id}]: physical motion confirmed")
            return VerificationResult(
                verification_status=VerificationStatus.VERIFIED,
                physical_motion_verified=True,
                reason="Physical locomotion confirmed by sensory telemetry.",
                expected_execution_status=expected_exec_status,
                expected_robot_state=expected_state,
                observed_robot_state=observed_state_dict,
            )
        else:
            # Invariant 1 & 5: Baseline current firmware without IMU/wheel encoders
            # Command was acknowledged and executed at controller level, but physical motion remains unverified
            logger.info(f"Verification NOT_VERIFIED for [{execution_record.execution_id}]: no sensory telemetry provided")
            return VerificationResult(
                verification_status=VerificationStatus.NOT_VERIFIED,
                physical_motion_verified=None,
                reason="Execution completed at controller level; physical motion unverified (no sensor telemetry provided by firmware).",
                expected_execution_status=expected_exec_status,
                expected_robot_state=expected_state,
                observed_robot_state=observed_state_dict,
            )

    # If observed motion state is unexpectedly still MOVING
    return VerificationResult(
        verification_status=VerificationStatus.FAILED,
        physical_motion_verified=None,
        reason=f"Unexpected final robot motion state after execution: {state_after.motion_state}.",
        expected_execution_status=expected_exec_status,
        expected_robot_state=expected_state,
        observed_robot_state=observed_state_dict,
    )
