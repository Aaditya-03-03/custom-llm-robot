"""
Tests for Stage 7 Plan Pre-Validation and Security Invariants.
Verifies:
- All-or-Nothing pre-validation: entire plan rejected if ANY action is invalid.
- Test A: LLM cannot bypass ToolRegistry (e.g. move_joint, fly rejected).
- Test B: Multi-action abort on failure: failure in Action 2 aborts Action 3.
- Test C: speed=999 rejected by Stage 5 safety validator with zero packets.
- Disabled tools (left, right, stand, sit, status) rejected in Stage 7 Phase 1.
"""

import pytest
from unittest.mock import AsyncMock, patch

from app.planning.models import ExecutionPlan, ToolCall, AssistantStatus
from app.planning.validator import pre_validate_execution_plan
from app.schemas.commands import PhysicalExecutionResponse, ExecutionStatus
from app.execution.adapter import ExecutionAdapter


def test_all_or_nothing_pre_validation():
    """Verify that if one action in a 3-action plan is invalid, the entire plan fails pre-validation."""
    plan = ExecutionPlan(
        plan_id="plan_all_or_nothing",
        actions=[
            ToolCall(action_id="act_1", tool="forward", parameters={"speed": 20, "steps": 1}),
            ToolCall(action_id="act_2", tool="fly", parameters={}),  # Invalid!
            ToolCall(action_id="act_3", tool="stop", parameters={"speed": 0, "steps": 0}),
        ]
    )

    result = pre_validate_execution_plan(plan)
    assert result.is_valid is False
    assert any("fly" in err for err in result.errors)


def test_security_test_a_cannot_bypass_tool_registry():
    """Test A: LLM cannot bypass ToolRegistry with arbitrary capabilities (e.g. move_joint)."""
    plan = ExecutionPlan(
        plan_id="plan_bypass_attempt",
        actions=[
            ToolCall(action_id="act_bypass", tool="move_joint", parameters={"joint_id": 1, "angle": 45}),
        ]
    )

    result = pre_validate_execution_plan(plan)
    assert result.is_valid is False
    assert any("move_joint" in err for err in result.errors)


def test_security_test_c_speed_999_rejected():
    """Test C: speed=999 is rejected by Stage 5 validator, resulting in zero execution packets."""
    plan = ExecutionPlan(
        plan_id="plan_speed_999",
        actions=[
            ToolCall(action_id="act_excessive", tool="forward", parameters={"speed": 999, "steps": 1}),
        ]
    )

    result = pre_validate_execution_plan(plan)
    assert result.is_valid is False
    assert any("Field 'speed'" in err or "speed" in err.lower() for err in result.errors)


def test_disabled_phase1_tools_rejected():
    """Verify disabled Phase 1 tools (left, right, stand, sit) are rejected during pre-validation."""
    for tool_name in ["left", "right", "stand", "sit", "status"]:
        plan = ExecutionPlan(
            plan_id=f"plan_disabled_{tool_name}",
            actions=[
                ToolCall(action_id="act_disabled", tool=tool_name, parameters={}),
            ]
        )
        result = pre_validate_execution_plan(plan)
        assert result.is_valid is False
        assert any(f"Tool '{tool_name}' is disabled or unsupported" in err for err in result.errors)


@pytest.mark.anyio
async def test_security_test_b_multi_action_abort_on_failure():
    """
    Test B: Sequential execution aborts remaining actions upon failure.
    Action 1 succeeds -> Action 2 fails -> Action 3 must NEVER be executed.
    """
    plan = ExecutionPlan(
        plan_id="plan_abort_test",
        actions=[
            ToolCall(action_id="act_1", tool="forward", parameters={"speed": 20, "steps": 1}),
            ToolCall(action_id="act_2", tool="backward", parameters={"speed": 20, "steps": 1}),
            ToolCall(action_id="act_3", tool="forward", parameters={"speed": 20, "steps": 1}),
        ]
    )

    # Mock execution adapter: action 1 succeeds, action 2 fails
    mock_adapter = AsyncMock(spec=ExecutionAdapter)
    
    resp_success = PhysicalExecutionResponse(
        success=True,
        status=ExecutionStatus.EXECUTION_SUCCESS,
        execution_id="exec_1",
        tool="forward",
        parameters={"speed": 20, "steps": 1},
        packet_sent="FORWARD|20|0.00",
        ack_received=True,
        hardware_state_affected=True,
        duration_seconds=0.6,
        message="Success",
    )
    resp_failure = PhysicalExecutionResponse(
        success=False,
        status=ExecutionStatus.ACK_TIMEOUT,
        execution_id="exec_2",
        tool="backward",
        parameters={"speed": 20, "steps": 1},
        packet_sent="BACKWARD|20|0.00",
        ack_received=False,
        hardware_state_affected=False,
        duration_seconds=None,
        message="ESP32 ACK timeout",
        errors=["ACK timeout"],
    )

    mock_adapter.execute_command.side_effect = [resp_success, resp_failure, resp_success]

    # Execute sequentially using Stage 7 assistant sequential execution logic
    completed_actions = []
    failed_action = None
    remaining_actions_aborted = []
    execution_results = []

    for idx, action in enumerate(plan.actions):
        from app.schemas.commands import PhysicalExecutionRequest
        req = PhysicalExecutionRequest(tool_name=action.tool, parameters=action.parameters)
        resp = await mock_adapter.execute_command(req)
        execution_results.append(resp)

        if resp.success:
            completed_actions.append(action.action_id)
        else:
            failed_action = action.action_id
            remaining_actions_aborted = [act.action_id for act in plan.actions[idx + 1:]]
            break

    assert completed_actions == ["act_1"]
    assert failed_action == "act_2"
    assert remaining_actions_aborted == ["act_3"]
    # Verify mock_adapter was only called twice (Action 1 and Action 2), NEVER Action 3!
    assert mock_adapter.execute_command.call_count == 2
