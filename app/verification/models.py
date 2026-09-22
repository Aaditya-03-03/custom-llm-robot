"""
Pydantic Models and Enums for Stage 9 Closed-Loop Execution, Verification & Recovery.
Standardizes execution lifecycle phases, deterministic verification models,
controlled recovery actions, and complete traceability audit records.
"""

from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from app.schemas.commands import ExecutionStatus


class ExecutionPhase(str, Enum):
    """
    Explicit lifecycle phases for closed-loop execution.
    Preserves separation between execution transactions, telemetry, and physical verification.
    """
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    AUTHORIZED = "AUTHORIZED"
    DISPATCHED = "DISPATCHED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    EXECUTING = "EXECUTING"  # Denotes active Stage 6 execution transaction; does NOT imply physical motion
    COMPLETED = "COMPLETED"
    VERIFYING = "VERIFYING"
    VERIFIED = "VERIFIED"    # Terminal phase when the expected execution outcome is verified at the available verification level
    NOT_VERIFIED = "NOT_VERIFIED"  # Terminal phase when execution succeeded but physical evidence is unavailable or verification disabled
    RECOVERY = "RECOVERY"    # Active during recovery execution
    FAILED = "FAILED"        # Terminal phase when execution, state, or recovery failed


class VerificationStatus(str, Enum):
    """
    Stage 9 proof outcome regarding resulting robot state.
    """
    VERIFIED = "VERIFIED"
    NOT_VERIFIED = "NOT_VERIFIED"
    PENDING = "PENDING"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class RecoveryAction(str, Enum):
    """
    Controlled recovery action policies.
    """
    NONE = "NONE"
    STOP = "STOP"
    ABORT_REMAINING = "ABORT_REMAINING"


class VerificationResult(BaseModel):
    """
    Result of evaluating expected vs observed robot state and physical telemetry.
    """
    verification_status: VerificationStatus = Field(
        ..., description="Stage 9 verification status"
    )
    physical_motion_verified: Optional[bool] = Field(
        default=None,
        description="True if physical evidence exists, False if refuted, None if insufficient evidence",
    )
    reason: str = Field(..., description="Explanation of verification outcome")
    expected_execution_status: ExecutionStatus = Field(
        default=ExecutionStatus.EXECUTION_SUCCESS,
        description="Expected Stage 6 execution status",
    )
    expected_robot_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Expected controller/robot state snapshot",
    )
    observed_robot_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Observed Stage 8 robot state snapshot",
    )


class ExecutionRecord(BaseModel):
    """
    Canonical record of a physical action's entire lifecycle from creation to verification/recovery.
    Full ID traceability: request_id -> session_id -> plan_id -> action_id -> execution_id.
    """
    execution_id: str = Field(..., description="Unique ID for this physical execution")
    request_id: str = Field(..., description="Original assistant request ID")
    session_id: str = Field(..., description="Conversation session ID")
    plan_id: str = Field(..., description="Execution plan ID")
    action_id: str = Field(..., description="Plan action ID")

    command: str = Field(..., description="Executed command verb (e.g. FORWARD, STOP)")
    speed: Optional[int] = Field(default=None, description="Requested speed percentage")
    steps: Optional[int] = Field(default=None, description="Requested steps count")

    execution_phase: ExecutionPhase = Field(
        default=ExecutionPhase.CREATED,
        description="Current lifecycle phase",
    )
    execution_status: ExecutionStatus = Field(
        default=ExecutionStatus.EXECUTION_SUCCESS,
        description="Authoritative Stage 6 execution status",
    )
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.PENDING,
        description="Stage 9 verification status",
    )
    physical_motion_verified: Optional[bool] = Field(
        default=None,
        description="Physical evidence status: True (confirmed), False (refuted), None (unknown/insufficient)",
    )

    expected_execution_status: ExecutionStatus = Field(
        default=ExecutionStatus.EXECUTION_SUCCESS,
        description="Expected Stage 6 execution result",
    )
    expected_robot_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Expected controller/telemetry state",
    )
    observed_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Snapshot of Stage 8 robot state after execution",
    )

    ack_received: bool = Field(
        default=False,
        description="True if command packet received acknowledgment from ESP32",
    )
    recovery_action: Optional[RecoveryAction] = Field(
        default=None,
        description="Recovery action invoked if failure occurred",
    )
    recovery_result: Optional[str] = Field(
        default=None,
        description="Outcome of recovery action",
    )

    started_at: datetime = Field(..., description="UTC timestamp when execution was initialized")
    verification_deadline: datetime = Field(
        ...,
        description="Mandatory observation window deadline: started_at + ROBOT_VERIFICATION_TIMEOUT_SECONDS",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when execution and verification completed",
    )
