"""
Robot State API Router for IOFT Humanoid Robot.
Exposes GET /api/v1/robot/state.
Provides read-only access to normalized robot operational and telemetry state.
Guarantees ZERO hardware execution packets sent.
"""

import logging
from fastapi import APIRouter, HTTPException, status

from app.state.models import RobotState
from app.state.manager import default_state_manager
from app.verification.models import ExecutionRecord
from app.verification.manager import default_execution_manager

logger = logging.getLogger("custom_llm_robot.api.robot")
router = APIRouter()


@router.get(
    "/robot/state",
    response_model=RobotState,
    summary="Get current robot operational and telemetry state",
    description=(
        "Returns a read-only snapshot of the robot's current connection status, "
        "control authority, requested command, physical observed motion state, "
        "and last received acknowledgments. Strictly read-only; dispatches zero hardware packets."
    ),
)
async def get_robot_state_endpoint() -> RobotState:
    logger.debug("Received request for /api/v1/robot/state")
    return default_state_manager.get_state()


@router.get(
    "/robot/execution/{execution_id}",
    response_model=ExecutionRecord,
    summary="Get closed-loop execution and verification status",
    description=(
        "Returns a read-only snapshot of a physical execution record, including lifecycle phase, "
        "Stage 6 execution status, Stage 9 verification status, physical motion verification, "
        "and recovery actions if any. Strictly read-only; dispatches zero hardware packets."
    ),
)
async def get_robot_execution_endpoint(execution_id: str) -> ExecutionRecord:
    logger.debug(f"Received request for /api/v1/robot/execution/{execution_id}")
    record = default_execution_manager.get_record(execution_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution record '{execution_id}' not found.",
        )
    return record
