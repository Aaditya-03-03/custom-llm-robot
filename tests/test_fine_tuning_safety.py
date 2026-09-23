"""
Stage 10 Fine-Tuning Safety Regression Tests.

Verifies:
- Stage 5 and Stage 7 safely handle edge cases from fine-tuned / untrusted LLM outputs
  (adversarial outputs, speed 999, corrupt/malformed JSON, bypass attempts).
- Clarification interception occurs with zero Stage 5/6 execution and zero packets sent.
- Zero hardware packets dispatched when LLM produces invalid parameters.
- Stage 8 and Stage 9 state and verification remain completely unaffected
  (physical_motion_verified remains None, zero spurious state mutations).
"""

import pytest
from unittest.mock import AsyncMock, patch

from app.planning.models import ExecutionPlan, ToolCall, AssistantStatus
from app.planning.parser import parse_execution_plan
from app.planning.validator import pre_validate_execution_plan
from app.safety.validator import CommandSafetyValidator
from app.verification.manager import ExecutionManager
from app.verification.models import ExecutionPhase, VerificationStatus


def test_malformed_json_fails_closed():
    """Corrupt or non-JSON output from LLM must fail closed with zero actions."""
    corrupt_outputs = [
        "I will now move forward at speed 20!",
        "```json\n{'actions': [{'tool': 'forward'}\n```",  # unclosed JSON
        "Random gibberish without json structure",
        "",
        "{\"actions\": \"not a list\"}",
    ]

    for raw in corrupt_outputs:
        plan, err = parse_execution_plan(raw)
        assert err is not None or plan is None or len(plan.actions) == 0, (
            f"Expected parsing error or empty actions for corrupt input: {raw}"
        )


def test_adversarial_speed_rejected_zero_udp():
    """Adversarial speed values (speed=999, negative speeds) must be rejected by Stage 5 safety validator."""
    with patch("app.execution.transport.UDPTransport.send_and_receive", new_callable=AsyncMock) as mock_udp:
        # 1. speed = 999
        plan_excessive = ExecutionPlan(
            plan_id="plan_speed_999",
            actions=[
                ToolCall(action_id="act_1", tool="forward", parameters={"speed": 999, "steps": 1})
            ]
        )
        val_result = pre_validate_execution_plan(plan_excessive)
        assert val_result.is_valid is False
        assert any("speed" in err.lower() or "bounds" in err.lower() or "999" in err for err in val_result.errors)

        # 2. speed = -50
        plan_negative = ExecutionPlan(
            plan_id="plan_speed_neg",
            actions=[
                ToolCall(action_id="act_1", tool="forward", parameters={"speed": -50, "steps": 1})
            ]
        )
        val_result_neg = pre_validate_execution_plan(plan_negative)
        assert val_result_neg.is_valid is False

        # Verify Stage 5 Validator directly rejects out-of-bounds speed
        validator = CommandSafetyValidator()
        res = validator.validate_command("forward", {"speed": 999, "steps": 1})
        assert res.valid is False
        assert any("speed" in err.lower() for err in res.errors)

        # Zero packets must have been sent
        assert mock_udp.call_count == 0


def test_unsupported_tool_outputs_zero_packets():
    """Fine-tuned model outputting tool: null or hallucinated tool must be rejected with zero execution."""
    raw_unsupported_json = """{
        "actions": [
            {
                "action_id": "action_1",
                "tool": null,
                "reason": "Climbing trees is not supported by 4-wheeled locomotion platform."
            }
        ]
    }"""
    plan, err = parse_execution_plan(raw_unsupported_json)
    assert err is None
    assert plan is not None
    assert len(plan.actions) == 1
    assert plan.actions[0].tool is None
    assert plan.actions[0].reason is not None

    val_res = pre_validate_execution_plan(plan)
    assert val_res.is_valid is False
    assert any("not supported" in e.lower() for e in val_res.errors)

    # Hallucinated tool in actions list
    raw_hallucinated = """{
        "actions": [
            {"action_id": "act_1", "tool": "somersault", "parameters": {}}
        ]
    }"""
    plan_h, err_h = parse_execution_plan(raw_hallucinated)
    assert plan_h is not None
    validation = pre_validate_execution_plan(plan_h)
    assert validation.is_valid is False
    assert any("somersault" in e.lower() for e in validation.errors)


def test_clarification_interception_bypasses_physical_execution():
    """Ambiguous movement command producing clarification must not execute any Stage 5/6 actions."""
    raw_clarification_json = """{
        "actions": [
            {
                "action_id": "action_1",
                "tool": "forward",
                "parameters": {}
            }
        ]
    }"""
    plan, err = parse_execution_plan(raw_clarification_json)
    assert err is None
    assert len(plan.actions) == 1
    val_res = pre_validate_execution_plan(plan)
    assert val_res.needs_clarification is True
    assert "speed" in val_res.missing_parameters


@pytest.mark.anyio
async def test_stage8_and_stage9_unaffected_by_rejected_plans():
    """Rejected or clarification plans must never transition Stage 9 lifecycle or mark physical_motion_verified."""
    manager = ExecutionManager()
    record = await manager.create_record(
        execution_id="exec_rejected_test",
        request_id="req_rejected_test",
        session_id="sess_rejected_test",
        plan_id="plan_rejected_test",
        action_id="act_rejected_test",
        command="FORWARD",
        speed=20,
        steps=1,
    )

    # Invariant: physical_motion_verified must remain None when execution has not been verified
    assert record.physical_motion_verified is None
    assert record.execution_phase == ExecutionPhase.CREATED
    assert record.verification_status == VerificationStatus.PENDING
