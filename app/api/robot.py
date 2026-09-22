"""
Robot State API Router for IOFT Humanoid Robot.
Exposes GET /api/v1/robot/state.
Provides read-only access to normalized robot operational and telemetry state.
Guarantees ZERO hardware execution packets sent.
"""

import logging
from fastapi import APIRouter

from app.state.models import RobotState
from app.state.manager import default_state_manager

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
