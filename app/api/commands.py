"""
Commands API Router for IOFT Humanoid Robot.
Exposes /commands/validate and /commands/dry-run under /api/v1.
Enforces fail-closed safety checks and dry-run simulations with zero hardware execution.
"""

import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status

from app.schemas.commands import (
    CommandValidationRequest,
    CommandValidationResponse,
    DryRunRequest,
    DryRunResponse,
    CommandExecutionResponse,
)
from app.safety.validator import CommandSafetyValidator
from app.commands.simulator import CommandSimulator

logger = logging.getLogger("custom_llm_robot.api.commands")
router = APIRouter()


@router.post(
    "/commands/validate",
    response_model=CommandValidationResponse,
    summary="Validate robot command against safety constraints",
    description=(
        "Evaluates a tool name and parameters against the authoritative CommandSafetyValidator. "
        "Fails closed on unknown capabilities, type mismatches, extra parameters, or range violations."
    ),
)
async def validate_command_endpoint(request: CommandValidationRequest):
    if not request.tool_name or not request.tool_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tool name cannot be empty or whitespace.",
        )

    logger.info(f"Received command validation request: tool='{request.tool_name}'")
    result = CommandSafetyValidator.validate_command(
        tool_name=request.tool_name,
        parameters=request.parameters,
    )
    return result


@router.post(
    "/commands/dry-run",
    response_model=DryRunResponse,
    summary="Simulate dry-run robot command execution",
    description=(
        "Simulates command execution in a safe dry-run environment. "
        "Guarantees hardware_state_affected=False and zero hardware execution. "
        "Internally invokes the authoritative CommandSafetyValidator pipeline."
    ),
)
async def dry_run_command_endpoint(request: DryRunRequest):
    if not request.tool_name or not request.tool_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tool name cannot be empty or whitespace.",
        )

    logger.info(f"Received command dry-run request: tool='{request.tool_name}'")
    result = CommandSimulator.simulate_dry_run(
        tool_name=request.tool_name,
        parameters=request.parameters,
    )
    return result


# Preserved legacy Stage 1 placeholder for backward compatibility
@router.post(
    "/commands",
    response_model=CommandExecutionResponse,
    summary="Legacy command endpoint placeholder",
    deprecated=True,
)
async def legacy_command_endpoint():
    """Legacy command endpoint placeholder. Direct execution deferred to Stage 6."""
    return CommandExecutionResponse(
        success=True,
        message="Direct command execution deferred to Stage 6. Use /commands/dry-run for simulation.",
        result={"status": "dry_run_recommended"},
    )
