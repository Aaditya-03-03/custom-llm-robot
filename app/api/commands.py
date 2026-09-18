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
    PhysicalExecutionRequest,
    PhysicalExecutionResponse,
    AuthorityStatusResponse,
    AuthorityUpdateRequest,
)
from app.core.config import ControlAuthority
from app.safety.validator import CommandSafetyValidator
from app.commands.simulator import CommandSimulator
from app.execution.adapter import default_execution_adapter
from app.execution.authority import default_authority_manager
from app.execution.errors import AuthorityConflictError

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
        message="Legacy placeholder. Direct execution implemented in Stage 6. Use /commands/execute for physical locomotion or /commands/dry-run for simulation.",
        result={"status": "use_commands_execute"},
    )


# ---------------------------------------------------------------------------
# Stage 6 Physical Execution & Authority Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/commands/execute",
    response_model=PhysicalExecutionResponse,
    summary="Execute physical robot locomotion command",
    description=(
        "Physically dispatches validated robot command to ESP32 over UDP port 8888. "
        "Enforces Stage 5 safety validation, AI execution authority check, "
        "mutual exclusion concurrency check, and confirmed MOTOR_ACK."
    ),
)
async def execute_command_endpoint(request: PhysicalExecutionRequest):
    if not request.tool_name or not request.tool_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tool name cannot be empty or whitespace.",
        )

    logger.info(f"Received physical execution request: tool='{request.tool_name}'")
    response = await default_execution_adapter.execute_command(request)
    return response


@router.get(
    "/commands/ownership",
    response_model=AuthorityStatusResponse,
    summary="Get current AI server execution authority",
    description="Query whether AI server execution authority is MANUAL or AI.",
)
async def get_ownership_endpoint():
    return default_authority_manager.status_dict


@router.post(
    "/commands/ownership",
    response_model=AuthorityStatusResponse,
    summary="Update AI server execution authority",
    description=(
        "Transition AI server authority between MANUAL and AI. "
        "On AI -> MANUAL: dispatches STOP|0|0.00 and verifies MOTOR_ACK before committing."
    ),
)
async def update_ownership_endpoint(request: AuthorityUpdateRequest):
    target = request.authority.upper().strip()
    if target not in (ControlAuthority.MANUAL.value, ControlAuthority.AI.value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid authority '{request.authority}'. Must be 'MANUAL' or 'AI'.",
        )

    target_enum = ControlAuthority(target)
    try:
        await default_authority_manager.set_authority(
            target_authority=target_enum,
            changed_by="api",
        )
        return default_authority_manager.status_dict
    except AuthorityConflictError as e:
        logger.error(f"Authority transition conflict: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )

