"""
Controlled Recovery Subsystem for Stage 9.
Defines deterministic recovery policies when physical execution, robot state,
or verification fails.
Enforces Invariants:
- Invariant 6: Recovery deduplication (avoids sending duplicate STOP if Stage 6 already issued it)
- Invariant 7: Recovery cannot bypass safety (routes strictly through Stage 5 & 6)
- Invariant 8: STOP is the primary recovery action; NO automatic movement retries
"""

import logging
from typing import Tuple

from app.core.config import settings
from app.execution.adapter import default_execution_adapter
from app.schemas.commands import ExecutionStatus, PhysicalExecutionRequest, PhysicalExecutionResponse
from app.verification.models import RecoveryAction, VerificationResult, VerificationStatus

logger = logging.getLogger("custom_llm_robot.verification.recovery")


def determine_recovery_policy(
    stage6_resp: PhysicalExecutionResponse,
    verification_result: VerificationResult,
    is_stale: bool = False,
) -> Tuple[RecoveryAction, bool]:
    """
    Determine the allowed controlled recovery action and whether Stage 6 already performed it.
    Returns:
        (RecoveryAction, already_performed: bool)
    """
    if not settings.ROBOT_RECOVERY_ENABLED:
        logger.info("Controlled recovery disabled by configuration.")
        return RecoveryAction.NONE, False

    # 1. Successful verification or unverified locomotion (normal completion with no sensor telemetry)
    # Lack of sensor telemetry is not an error requiring emergency STOP
    if verification_result.verification_status in (
        VerificationStatus.VERIFIED,
        VerificationStatus.NOT_VERIFIED,
    ):
        return RecoveryAction.NONE, False

    # 2. Stage 6 ACK Timeout or Execution Failure
    # Check if Stage 6 execution layer already issued an emergency STOP failsafe
    if stage6_resp.status == ExecutionStatus.ACK_TIMEOUT:
        # If hardware state was affected, Stage 6 timer / pulse already issued emergency STOP
        if stage6_resp.hardware_state_affected:
            logger.info("Recovery: Stage 6 already issued emergency STOP failsafe during ACK_TIMEOUT.")
            return RecoveryAction.STOP, True
        else:
            # Command failed before packets affected hardware, but controlled STOP is safest
            logger.info("Recovery: Stage 6 ACK_TIMEOUT before hardware effect; Stage 9 STOP recommended.")
            return RecoveryAction.STOP, False

    # 3. State Mismatch / Stale Heartbeat / Sensory Motion Refuted
    if is_stale or verification_result.verification_status == VerificationStatus.FAILED:
        logger.warning(
            f"Recovery: Verification FAILED or robot stale (is_stale={is_stale}, "
            f"reason='{verification_result.reason}'). Controlled STOP required."
        )
        return RecoveryAction.STOP, False

    return RecoveryAction.NONE, False


async def execute_controlled_recovery(recovery_action: RecoveryAction) -> str:
    """
    Execute controlled recovery action strictly through Stage 5 & Stage 6.
    Zero raw UDP sockets are opened here.
    """
    if recovery_action == RecoveryAction.NONE:
        return "No recovery action required."

    if recovery_action == RecoveryAction.ABORT_REMAINING:
        return "Subsequent actions safely aborted."

    if recovery_action == RecoveryAction.STOP:
        logger.info("Executing controlled recovery: dispatching STOP via Stage 6 execution adapter.")
        try:
            req = PhysicalExecutionRequest(tool_name="stop", parameters={})
            resp = await default_execution_adapter.execute_command(req)
            if resp.success:
                return f"Controlled recovery STOP successfully executed: {resp.message}"
            else:
                return f"Controlled recovery STOP dispatched but unacknowledged: {resp.message}"
        except Exception as e:
            logger.error(f"Error during controlled recovery STOP dispatch: {e}")
            return f"Controlled recovery STOP dispatch error: {str(e)}"

    return f"Unknown recovery action: {recovery_action}"
