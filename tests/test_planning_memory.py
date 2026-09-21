"""
Tests for Stage 7 Contextual Memory and Reference Resolution.
Verifies:
- 'do that again' inherits tool and parameters from previous action context.
- 'do that again, but twice as long' scales the steps parameter.
- Safety validation is strictly enforced on memory-derived actions.
"""

import pytest
from app.planning.parser import parse_action, resolve_memory_reference
from app.planning.validator import pre_validate_execution_plan
from app.planning.models import ExecutionPlan, ToolCall


def test_resolve_memory_reference_simple_repeat():
    """Verify 'do that again' inherits previous tool and parameters."""
    recent_context = {
        "tool": "forward",
        "parameters": {"speed": 20, "steps": 1}
    }
    proposal = {"action_id": "act_1", "tool": "repeat", "parameters": {}}
    user_msg = "do that again"

    resolved = resolve_memory_reference(proposal, recent_context=recent_context, user_message=user_msg)
    assert resolved["tool"] == "forward"
    assert resolved["parameters"]["speed"] == 20
    assert resolved["parameters"]["steps"] == 1


def test_resolve_memory_reference_twice_as_long():
    """Verify 'do that again, but twice as long' doubles previous steps."""
    recent_context = {
        "tool": "backward",
        "parameters": {"speed": 40, "steps": 2}
    }
    proposal = {"action_id": "act_1", "tool": "repeat", "parameters": {}}
    user_msg = "do that again, but twice as long"

    resolved = resolve_memory_reference(proposal, recent_context=recent_context, user_message=user_msg)
    assert resolved["tool"] == "backward"
    assert resolved["parameters"]["speed"] == 40
    assert resolved["parameters"]["steps"] == 4


def test_memory_derived_action_still_validated():
    """Verify memory-derived actions undergo Stage 5 safety validation."""
    recent_context = {
        "tool": "forward",
        "parameters": {"speed": 999, "steps": 1}  # Invalid speed in corrupted context
    }
    action = parse_action(
        {"action_id": "act_1", "tool": "repeat", "parameters": {}},
        recent_context=recent_context,
        user_message="repeat that"
    )
    plan = ExecutionPlan(plan_id="plan_mem_val", actions=[action])
    result = pre_validate_execution_plan(plan)
    assert result.is_valid is False
    assert any("speed" in err.lower() for err in result.errors)
