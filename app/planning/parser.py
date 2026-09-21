"""
Deterministic Parser & Parameter Resolution for Stage 7 Action Planning.
Converts LLM tool proposals into strongly typed ToolCall and ExecutionPlan models.
"""

import json
import re
import uuid
from typing import Dict, Any, Optional, List, Tuple

from app.core.config import settings
from app.planning.models import ToolCall, ExecutionPlan
from app.tools.definitions import is_tool_enabled


SPEED_PROFILES: Dict[str, int] = {
    "slow": settings.ROBOT_SPEED_SLOW,
    "medium": settings.ROBOT_SPEED_MEDIUM,
    "fast": settings.ROBOT_SPEED_FAST,
}

MOVEMENT_TOOLS = {"forward", "backward", "left", "right"}


class PlanParsingError(Exception):
    """Raised when an LLM plan is malformed or exceeds constraints."""
    pass


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from raw LLM text output.
    Handles Markdown code fences (```json ... ```) or bare JSON.
    """
    text = text.strip()
    # Match markdown fence
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Match outermost curly braces
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            json_str = text[brace_start : brace_end + 1]
        else:
            json_str = text

    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise PlanParsingError("LLM response must be a JSON object")
        return data
    except (json.JSONDecodeError, Exception) as e:
        raise PlanParsingError(f"Failed to parse LLM response as JSON: {e}")


def resolve_memory_reference(
    tool_proposal: Dict[str, Any],
    recent_context: Optional[Dict[str, Any]] = None,
    user_message: str = "",
) -> Dict[str, Any]:
    """
    If the user command is a contextual reference (e.g., 'do that again', 'repeat'),
    inherit tool parameters from the last successfully executed action in recent_context.
    """
    if not recent_context:
        return tool_proposal

    msg_lower = user_message.lower()
    is_repetition = any(phrase in msg_lower for phrase in ["do that again", "repeat", "once more", "again", "same thing"])

    if is_repetition and ("tool" not in tool_proposal or tool_proposal.get("tool") is None or tool_proposal.get("tool") == "repeat"):
        last_tool = recent_context.get("tool")
        last_params = recent_context.get("parameters", {})
        if last_tool:
            resolved = dict(tool_proposal)
            resolved["tool"] = last_tool
            params = dict(last_params)
            
            # Check for modifiers like "twice as long" or "for 2 steps"
            if "twice" in msg_lower or "double" in msg_lower:
                params["steps"] = int(last_params.get("steps", 1)) * 2
            elif "three times" in msg_lower:
                params["steps"] = int(last_params.get("steps", 1)) * 3
            
            resolved["parameters"] = params
            return resolved

    return tool_proposal


def parse_action(
    raw_action: Dict[str, Any],
    action_index: int = 1,
    recent_context: Optional[Dict[str, Any]] = None,
    user_message: str = "",
) -> ToolCall:
    """
    Parse and deterministically resolve a single tool proposal into a ToolCall.
    """
    action_id = raw_action.get("action_id", f"action_{action_index}")
    
    # Resolve memory references if needed
    raw_action = resolve_memory_reference(raw_action, recent_context=recent_context, user_message=user_message)

    tool = raw_action.get("tool")
    if not tool:
        return ToolCall(
            action_id=action_id,
            tool=None,
            reason=raw_action.get("reason", "No tool specified by LLM"),
        )

    canonical_tool = str(tool).lower().strip()
    raw_params = raw_action.get("parameters") or {}
    if not isinstance(raw_params, dict):
        return ToolCall(
            action_id=action_id,
            tool=canonical_tool,
            reason="Action parameters must be a dictionary",
        )

    # 1. Stop tool requires no speed
    if canonical_tool == "stop":
        return ToolCall(
            action_id=action_id,
            tool="stop",
            parameters={"speed": 0, "steps": 0},
            steps=0,
            requested_speed=0,
        )

    # 2. Movement tools: forward, backward, left, right
    if canonical_tool in MOVEMENT_TOOLS:
        has_speed = "speed" in raw_params and raw_params["speed"] is not None
        has_profile = "speed_profile" in raw_params and raw_params["speed_profile"] is not None

        # Contradiction check: cannot specify both speed and speed_profile
        if has_speed and has_profile:
            return ToolCall(
                action_id=action_id,
                tool=canonical_tool,
                reason="Cannot specify both 'speed' and 'speed_profile'. Specify one or the other.",
            )

        resolved_speed: Optional[int] = None
        speed_profile_val: Optional[str] = None
        requested_speed_val: Optional[int] = None

        if has_profile:
            profile_str = str(raw_params["speed_profile"]).lower().strip()
            if profile_str not in SPEED_PROFILES:
                return ToolCall(
                    action_id=action_id,
                    tool=canonical_tool,
                    reason=f"Invalid speed_profile: '{profile_str}'. Must be one of {list(SPEED_PROFILES.keys())}.",
                )
            speed_profile_val = profile_str
            resolved_speed = SPEED_PROFILES[profile_str]
        elif has_speed:
            try:
                requested_speed_val = int(raw_params["speed"])
                resolved_speed = requested_speed_val
            except (ValueError, TypeError):
                return ToolCall(
                    action_id=action_id,
                    tool=canonical_tool,
                    reason="Invalid speed value. Must be an integer.",
                )
        else:
            # Missing required parameter: speed is missing!
            # Distinguished: Missing parameter -> needs_clarification
            return ToolCall(
                action_id=action_id,
                tool=canonical_tool,
                needs_clarification=True,
                missing=["speed"],
                reason="Movement command requires speed or speed_profile",
            )

        # Steps resolution: defaults to 1 if omitted
        steps_val = 1
        if "steps" in raw_params and raw_params["steps"] is not None:
            try:
                steps_val = int(raw_params["steps"])
            except (ValueError, TypeError):
                return ToolCall(
                    action_id=action_id,
                    tool=canonical_tool,
                    reason="Invalid steps value. Must be an integer.",
                )

        resolved_params: Dict[str, Any] = {
            "speed": resolved_speed,
            "steps": steps_val,
        }

        # Include optional duration if present
        if "duration" in raw_params and raw_params["duration"] is not None:
            try:
                resolved_params["duration"] = float(raw_params["duration"])
            except (ValueError, TypeError):
                pass

        return ToolCall(
            action_id=action_id,
            tool=canonical_tool,
            parameters=resolved_params,
            speed_profile=speed_profile_val,
            requested_speed=requested_speed_val if requested_speed_val is not None else resolved_speed,
            steps=steps_val,
        )

    # 3. Other tools (e.g. stand, sit, status, or arbitrary tool calls)
    # Pass parameters as provided for validation by ToolRegistry/CommandSafetyValidator
    return ToolCall(
        action_id=action_id,
        tool=canonical_tool,
        parameters=raw_params,
    )


def parse_execution_plan(
    raw_response: str | Dict[str, Any],
    recent_context: Optional[Dict[str, Any]] = None,
    user_message: str = "",
) -> Tuple[ExecutionPlan, Optional[str]]:
    """
    Parses full LLM output into an ExecutionPlan.
    Enforces MAX_ACTIONS_PER_PLAN constraint.
    Returns (ExecutionPlan, error_message).
    """
    if isinstance(raw_response, str):
        try:
            data = extract_json_from_text(raw_response)
        except PlanParsingError as e:
            return ExecutionPlan(
                plan_id=str(uuid.uuid4()),
                actions=[],
                plan_explanation=None,
            ), str(e)
    elif isinstance(raw_response, dict):
        data = raw_response
    else:
        return ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            actions=[],
        ), "Invalid response format"

    plan_explanation = data.get("plan_explanation") or data.get("explanation")
    raw_actions = data.get("actions", [])

    # If single tool call is at top level
    if not raw_actions and "tool" in data:
        raw_actions = [{
            "action_id": "action_1",
            "tool": data.get("tool"),
            "parameters": data.get("parameters", {}),
        }]

    if not isinstance(raw_actions, list):
        return ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            actions=[],
            plan_explanation=plan_explanation,
        ), "'actions' field must be a list"

    # Enforce maximum actions per plan limit
    if len(raw_actions) > settings.MAX_ACTIONS_PER_PLAN:
        return ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            actions=[],
            plan_explanation=plan_explanation,
        ), f"Plan exceeds maximum allowed actions ({settings.MAX_ACTIONS_PER_PLAN}). Got {len(raw_actions)}."

    parsed_actions: List[ToolCall] = []
    for idx, raw_act in enumerate(raw_actions, start=1):
        if not isinstance(raw_act, dict):
            return ExecutionPlan(
                plan_id=str(uuid.uuid4()),
                actions=[],
                plan_explanation=plan_explanation,
            ), f"Action at index {idx} must be an object"
        
        parsed_call = parse_action(
            raw_act,
            action_index=idx,
            recent_context=recent_context,
            user_message=user_message,
        )
        parsed_actions.append(parsed_call)

    plan = ExecutionPlan(
        plan_id=str(uuid.uuid4()),
        actions=parsed_actions,
        plan_explanation=plan_explanation,
    )
    return plan, None
