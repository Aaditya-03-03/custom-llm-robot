"""
All-or-Nothing Plan Pre-Validator for Stage 7 Action Planning.
Pre-validates all actions in an ExecutionPlan against the ToolRegistry, runtime tool enablement,
and Stage 5 CommandSafetyValidator before any execution is permitted.
"""

import logging
from typing import Tuple, List, Optional

from app.planning.models import ExecutionPlan, ToolCall
from app.tools.definitions import is_tool_enabled
from app.tools.registry import default_tool_registry
from app.safety.validator import CommandSafetyValidator

logger = logging.getLogger("custom_llm_robot.planning.validator")


class PlanValidationResult:
    def __init__(
        self,
        is_valid: bool,
        needs_clarification: bool = False,
        missing_parameters: Optional[List[str]] = None,
        errors: Optional[List[str]] = None,
        validated_plan: Optional[ExecutionPlan] = None,
    ):
        self.is_valid = is_valid
        self.needs_clarification = needs_clarification
        self.missing_parameters = missing_parameters or []
        self.errors = errors or []
        self.validated_plan = validated_plan


def pre_validate_execution_plan(plan: ExecutionPlan) -> PlanValidationResult:
    """
    Validates all actions in an ExecutionPlan.
    Enforces all-or-nothing: if ANY action is invalid or unsupported,
    the entire plan is rejected before physical execution.
    
    Distinguishes:
      - Missing required parameters (e.g. speed missing) -> needs_clarification=True (no Stage 5/6 calls)
      - Invalid parameters (e.g. speed=999) -> is_valid=False (Stage 5 rejects)
    """
    if not plan.actions:
        return PlanValidationResult(
            is_valid=False,
            errors=["Execution plan contains no executable actions."],
            validated_plan=plan,
        )

    # 1. Check if any action requires clarification (missing required parameter)
    for action in plan.actions:
        if action.needs_clarification:
            logger.info(f"Action '{action.tool}' requires clarification: missing {action.missing}")
            return PlanValidationResult(
                is_valid=False,
                needs_clarification=True,
                missing_parameters=action.missing,
                errors=[action.reason or "Missing required parameter"],
                validated_plan=plan,
            )

    all_errors: List[str] = []

    # 2. Validate every action against Registry, Enablement, and SafetyValidator
    for action in plan.actions:
        tool_name = action.tool
        if not tool_name:
            all_errors.append(f"Action '{action.action_id}': {action.reason or 'No tool specified'}")
            continue

        # Check parser-level reason / conflicting parameters
        if action.reason:
            all_errors.append(f"Action '{action.action_id}' ({tool_name}): {action.reason}")
            continue

        # Check runtime tool enablement (Stage 7 Phase 1: forward, backward, stop enabled)
        if not is_tool_enabled(tool_name):
            all_errors.append(
                f"Action '{action.action_id}': Tool '{tool_name}' is disabled or unsupported in Stage 7."
            )
            continue

        # Check closed ToolRegistry
        if not default_tool_registry.is_registered(tool_name):
            all_errors.append(
                f"Action '{action.action_id}': Tool '{tool_name}' is not in the allowlisted tool registry."
            )
            continue

        # Call Stage 5 authoritative safety validator
        validation_resp = CommandSafetyValidator.validate_command(
            tool_name=tool_name,
            parameters=action.parameters,
            registry=default_tool_registry,
        )

        if not validation_resp.valid:
            logger.warning(f"Action '{action.action_id}' failed safety validation: {validation_resp.errors}")
            all_errors.extend([f"Action '{action.action_id}': {err}" for err in validation_resp.errors])
        else:
            # Update with clean parameters from validator
            action.parameters = validation_resp.parameters

    if all_errors:
        logger.warning(f"Plan '{plan.plan_id}' rejected during pre-validation: {all_errors}")
        return PlanValidationResult(
            is_valid=False,
            needs_clarification=False,
            errors=all_errors,
            validated_plan=plan,
        )

    logger.info(f"Plan '{plan.plan_id}' successfully pre-validated ({len(plan.actions)} actions)")
    return PlanValidationResult(
        is_valid=True,
        needs_clarification=False,
        errors=[],
        validated_plan=plan,
    )
