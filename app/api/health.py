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

    Returns the status of the LLM provider (Ollama) and the MongoDB
    conversation memory store. The server may be healthy overall even
    when MongoDB is temporarily unavailable, but chat will return 503
    in that state.
    """
    provider = get_llm_provider()
    llm_ok = await provider.check_health()
    mongo_ok = is_mongodb_available()

    logger.info(
        f"Health check: llm_available={llm_ok}, mongodb_available={mongo_ok}"
    )

    return HealthResponse(
        status="ok",
        llm_provider=settings.LLM_PROVIDER,
        model=settings.OLLAMA_MODEL,
        llm_available=llm_ok,
        mongodb_available=mongo_ok,
    )
