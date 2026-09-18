"""
Dedicated Intent Detection API Endpoint — POST /api/v1/intent.
Primary and authoritative endpoint for converting natural language robot requests
into validated structured intents.
"""

import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status

from app.intent.schemas import IntentRequest, StructuredIntent, IntentAuditRecord
from app.intent.extractor import IntentExtractor

logger = logging.getLogger("custom_llm_robot.api.intent")
router = APIRouter()


@router.post(
    "/intent",
    response_model=StructuredIntent,
    summary="Extract structured robot intent from natural language",
    description=(
        "Primary endpoint for intent detection. Converts natural language instructions "
        "into strict, machine-readable structured intents matching established robot capabilities. "
        "Interpretation-only: does not execute physical robot commands."
    ),
)
async def extract_intent_endpoint(request: IntentRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Instruction text cannot be empty or whitespace.",
        )

    logger.info(f"Received intent detection request: '{request.text[:80]}'")
    extractor = IntentExtractor()
    result = await extractor.extract_intent(request.text)

    # In-memory audit logging (decoupled from runtime DB persistence)
    audit_record = IntentAuditRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        raw_input=request.text,
        detected_category=result.category.value,
        detected_action=result.action.value if result.action else None,
        parameters=result.parameters.model_dump() if result.parameters else None,
        is_valid=result.is_valid,
    )
    logger.info(
        f"Intent detection completed: category={audit_record.detected_category}, "
        f"action={audit_record.detected_action}, valid={audit_record.is_valid}"
    )

    return result
