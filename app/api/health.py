"""
Health check endpoint — GET /api/v1/health

Reports:
- General server status
- Active LLM provider and model
- LLM availability (Ollama reachable?)
- MongoDB availability (conversation memory reachable?)
"""

import logging
from fastapi import APIRouter
from app.schemas.health import HealthResponse
from app.llm.factory import get_llm_provider
from app.memory.database import is_mongodb_available
from app.state import default_state_manager
from app.core.config import settings

logger = logging.getLogger("custom_llm_robot.api.health")
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
)
async def health_check():
    """
    Check the health of all system components.

    Returns the status of the LLM provider (Ollama), the MongoDB
    conversation memory store, and the robot state subsystem.
    """
    provider = get_llm_provider()
    llm_ok = await provider.check_health()
    mongo_ok = is_mongodb_available()
    robot_ok = default_state_manager.is_connected()
    state_subsystem_ok = default_state_manager is not None

    logger.info(
        f"Health check: llm_available={llm_ok}, mongodb_available={mongo_ok}, "
        f"robot_connected={robot_ok}, robot_state_available={state_subsystem_ok}"
    )

    return HealthResponse(
        status="ok",
        llm_provider=settings.LLM_PROVIDER,
        model=settings.OLLAMA_MODEL,
        llm_available=llm_ok,
        mongodb_available=mongo_ok,
        robot_connected=robot_ok,
        robot_state_available=state_subsystem_ok,
    )
