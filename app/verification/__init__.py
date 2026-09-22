"""
Stage 9 Verification & Recovery Package.
Exports core models, deterministic verifier, controlled recovery policies,
and the singleton ExecutionManager.
"""

from app.verification.models import (
    ExecutionPhase,
    VerificationStatus,
    RecoveryAction,
    VerificationResult,
    ExecutionRecord,
)
from app.verification.verifier import (
    get_expected_robot_state,
    verify_execution,
)
from app.verification.recovery import (
    determine_recovery_policy,
    execute_controlled_recovery,
)
from app.verification.manager import (
    ExecutionManager,
    default_execution_manager,
)

__all__ = [
    "ExecutionPhase",
    "VerificationStatus",
    "RecoveryAction",
    "VerificationResult",
    "ExecutionRecord",
    "get_expected_robot_state",
    "verify_execution",
    "determine_recovery_policy",
    "execute_controlled_recovery",
    "ExecutionManager",
    "default_execution_manager",
]
