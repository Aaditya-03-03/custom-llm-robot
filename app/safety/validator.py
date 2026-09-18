"""
Single authoritative safety validator for robot commands and hardware constraints.
Enforces fail-closed validation, exact strict types, and rejection of unknown parameters.
"""

import logging
from typing import Dict, Any, Optional, List
from pydantic import ValidationError

from app.safety.limits import JOINT_LIMITS
from app.tools.registry import ToolRegistry, default_tool_registry
from app.schemas.commands import (
    CommandValidationResponse,
    StrictEmptyParameters,
)

logger = logging.getLogger("custom_llm_robot.safety.validator")


class CommandSafetyValidator:
    """
    Authoritative safety validator for robot commands.
    Enforces fail-closed rules:
      - Capability allowlist checking via ToolRegistry
      - Strict parameter type validation (no coercion)
      - Explicit rejection of unknown/extra parameters
      - Value range checking (1 <= speed <= 100, steps > 0)
    """

    @staticmethod
    def validate_joint_movement(joint_id: int, angle: float) -> tuple[bool, str]:
        """Validate if target joint angle is within safety bounds (preserved utility)."""
        if joint_id not in JOINT_LIMITS:
            return False, f"Unknown joint ID {joint_id}"
        min_angle, max_angle = JOINT_LIMITS[joint_id]
        if not (min_angle <= angle <= max_angle):
            return False, f"Angle {angle} out of bounds [{min_angle}, {max_angle}] for joint {joint_id}"
        return True, "Valid"

    @classmethod
    def validate_command(
        cls,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        registry: Optional[ToolRegistry] = None,
    ) -> CommandValidationResponse:
        """
        Single authoritative validation pipeline for robot commands.
        Used identically by /commands/validate and /commands/dry-run.
        """
        reg = registry or default_tool_registry
        raw_params = parameters if parameters is not None else {}

        if not isinstance(raw_params, dict):
            logger.warning(f"Command validation failed: parameters must be a dictionary, got {type(raw_params)}")
            return CommandValidationResponse(
                valid=False,
                tool=tool_name or "unknown",
                parameters={},
                message="Rejected: Parameters must be a JSON object.",
                errors=["Invalid parameters type; expected dict/object."],
            )

        # 1. Capability Allowlist Resolution
        tool = reg.get_tool(tool_name)
        if tool is None:
            logger.warning(f"Command validation rejected unknown tool: '{tool_name}'")
            return CommandValidationResponse(
                valid=False,
                tool=tool_name or "unknown",
                parameters=raw_params,
                message=f"Rejected: Tool '{tool_name}' is not in the allowlisted tool registry.",
                errors=[f"Unknown or unsupported capability '{tool_name}'."],
            )

        # 2. Parameter Schema & Strictness Validation
        clean_parameters: Dict[str, Any] = {}
        errors: List[str] = []

        if tool.parameter_schema is None or tool.parameter_schema is StrictEmptyParameters:
            # Command must not receive any parameters (stand, sit, status, calibrate, manual)
            if raw_params:
                logger.warning(f"Command validation rejected unexpected parameters for '{tool.name}': {list(raw_params.keys())}")
                return CommandValidationResponse(
                    valid=False,
                    tool=tool.name,
                    parameters=raw_params,
                    message=f"Rejected: Tool '{tool.name}' does not accept parameters.",
                    errors=[f"Unexpected parameter(s) provided: {list(raw_params.keys())}"],
                )
        else:
            # Validate against strict schema (extra='forbid', strict=True)
            try:
                validated_model = tool.parameter_schema.model_validate(raw_params)
                clean_parameters = validated_model.model_dump(exclude_none=True)
            except ValidationError as val_err:
                for err in val_err.errors():
                    field_loc = " -> ".join(str(loc) for loc in err.get("loc", []))
                    err_msg = err.get("msg", "Invalid value")
                    err_type = err.get("type", "")
                    errors.append(f"Field '{field_loc}': {err_msg} ({err_type})")

                logger.warning(f"Command validation failed for '{tool.name}': {errors}")
                return CommandValidationResponse(
                    valid=False,
                    tool=tool.name,
                    parameters=raw_params,
                    message=f"Rejected: Parameter validation failed for tool '{tool.name}'.",
                    errors=errors,
                )

        logger.info(f"Command '{tool.name}' passed safety validation: {clean_parameters}")
        return CommandValidationResponse(
            valid=True,
            tool=tool.name,
            parameters=clean_parameters,
            message=f"Command '{tool.name}' validated successfully.",
            errors=[],
        )
