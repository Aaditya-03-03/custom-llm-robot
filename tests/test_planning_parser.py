"""
Tests for Stage 7 Plan Parser and Deterministic Parameter Resolution.
Verifies:
- Deterministic mapping: slow->20, medium->40, fast->70.
- Steps defaults to 1 if omitted.
- Rejection of conflicting parameters (both speed AND speed_profile).
- Rejection of unknown speed profiles (e.g. super_fast).
- Missing speed detection (needs_clarification=True).
- Enforcement of MAX_ACTIONS_PER_PLAN <= 3.
- Parsing from Markdown code blocks and handling of malformed inputs.
"""

import json
import pytest

from app.core.config import settings
from app.planning.parser import (
    parse_action,
    parse_execution_plan,
    extract_json_from_text,
    PlanParsingError,
)
from app.planning.models import ToolCall, ExecutionPlan


def test_extract_json_from_code_blocks():
    """Verify JSON extraction from markdown code fences."""
    raw = """
    Here is the proposed action plan:
    ```json
    {
      "plan_explanation": "Move forward slowly",
      "actions": [
        {
          "action_id": "action_1",
          "tool": "forward",
          "parameters": {
            "speed_profile": "slow",
            "steps": 1
          }
        }
      ]
    }
    ```
    Hope this helps!
    """
    data = extract_json_from_text(raw)
    assert data["plan_explanation"] == "Move forward slowly"
    assert len(data["actions"]) == 1


def test_deterministic_speed_profile_resolution():
    """Verify semantic speed profiles map deterministically to config values."""
    # slow
    call_slow = parse_action({
        "action_id": "act_1",
        "tool": "forward",
        "parameters": {"speed_profile": "slow"}
    })
    assert call_slow.parameters["speed"] == settings.ROBOT_SPEED_SLOW
    assert call_slow.parameters["steps"] == 1  # defaulted
    assert call_slow.speed_profile == "slow"
    assert call_slow.needs_clarification is False

    # medium
    call_med = parse_action({
        "action_id": "act_2",
        "tool": "backward",
        "parameters": {"speed_profile": "medium", "steps": 3}
    })
    assert call_med.parameters["speed"] == settings.ROBOT_SPEED_MEDIUM
    assert call_med.parameters["steps"] == 3
    assert call_med.speed_profile == "medium"

    # fast
    call_fast = parse_action({
        "action_id": "act_3",
        "tool": "forward",
        "parameters": {"speed_profile": "fast"}
    })
    assert call_fast.parameters["speed"] == settings.ROBOT_SPEED_FAST
    assert call_fast.parameters["steps"] == 1


def test_conflicting_speed_and_profile_rejection():
    """Verify providing BOTH numeric speed AND semantic speed_profile is rejected."""
    raw_action = {
        "action_id": "act_conflict",
        "tool": "forward",
        "parameters": {
            "speed": 50,
            "speed_profile": "slow",
            "steps": 1
        }
    }
    call = parse_action(raw_action)
    assert call.reason is not None
    assert "Cannot specify both 'speed' and 'speed_profile'" in call.reason


def test_unknown_speed_profile_rejection():
    """Verify unsupported speed profile strings are rejected."""
    raw_action = {
        "action_id": "act_unknown",
        "tool": "forward",
        "parameters": {
            "speed_profile": "super_fast",
            "steps": 1
        }
    }
    call = parse_action(raw_action)
    assert call.reason is not None
    assert "Invalid speed_profile" in call.reason


def test_missing_speed_clarification():
    """Verify omission of speed triggers needs_clarification=True."""
    raw_action = {
        "action_id": "act_missing",
        "tool": "forward",
        "parameters": {"steps": 1}
    }
    call = parse_action(raw_action)
    assert call.needs_clarification is True
    assert "speed" in call.missing


def test_stop_tool_requires_no_speed():
    """Verify stop tool needs no speed and does not require clarification."""
    raw_action = {
        "action_id": "act_stop",
        "tool": "stop",
        "parameters": {}
    }
    call = parse_action(raw_action)
    assert call.needs_clarification is False
    assert call.parameters["speed"] == 0
    assert call.parameters["steps"] == 0


def test_max_actions_per_plan_enforcement():
    """Verify plans exceeding MAX_ACTIONS_PER_PLAN (3) are rejected."""
    actions_4 = [
        {"action_id": f"act_{i}", "tool": "forward", "parameters": {"speed_profile": "slow"}}
        for i in range(1, 5)
    ]
    raw_plan = {
        "plan_explanation": "4 actions",
        "actions": actions_4
    }
    plan, err = parse_execution_plan(raw_plan)
    assert err is not None
    assert "exceeds maximum allowed actions" in err
    assert len(plan.actions) == 0


def test_malformed_json_handling():
    """Verify malformed JSON returns a clean error."""
    raw_text = "This is not json at all {corrupt"
    plan, err = parse_execution_plan(raw_text)
    assert err is not None
    assert "Failed to parse LLM response as JSON" in err
