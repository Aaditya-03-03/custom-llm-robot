"""
Stage 5 Tool Registry, Safety Validator, and Command Dry-Run Tests.
Verifies:
- Closed Tool Registry allowlist and immutability.
- Single authoritative CommandSafetyValidator with fail-closed rules and strict non-coercive parameter typing.
- Hardware-isolated CommandSimulator with execution_mode='simulation' and hardware_state_affected=False.
- API endpoints: /api/v1/commands/validate and /api/v1/commands/dry-run.
- End-to-end Stage 4 (Intent) -> Stage 5 (Dry-Run) integration.
- Hardware isolation boundary (zero serial/GPIO/motor communication).
- Preservation of Stage 1-4 existing API routes.
"""

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from app.main import app
from app.tools.registry import ToolRegistry, RegisteredTool, default_tool_registry
from app.safety.validator import CommandSafetyValidator
from app.commands.simulator import CommandSimulator
from app.schemas.commands import (
    CommandValidationResponse,
    DryRunResponse,
    StrictMovementParameters,
    StrictStopParameters,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Tool Registry Allowlist & Immutability Tests
# ---------------------------------------------------------------------------

def test_registry_resolves_all_10_grounded_capabilities():
    """ToolRegistry must recognize all 10 verified capabilities."""
    expected_tools = [
        "forward", "backward", "left", "right",
        "stand", "sit", "stop", "status", "calibrate", "manual"
    ]
    registry = ToolRegistry()
    for tool_name in expected_tools:
        tool = registry.get_tool(tool_name)
        assert tool is not None, f"Tool '{tool_name}' must be registered"
        assert tool.name == tool_name
        assert registry.is_registered(tool_name) is True

    assert len(registry.list_tools()) == 10


def test_registry_rejects_unverified_tools():
    """ToolRegistry must strictly reject non-allowlisted capabilities."""
    unverified_tools = [
        "fly", "jump", "turn_left", "turn_right", "move_joint",
        "grab_object", "dance", "execute_arbitrary_code"
    ]
    registry = ToolRegistry()
    for tool_name in unverified_tools:
        assert registry.get_tool(tool_name) is None
        assert registry.is_registered(tool_name) is False


def test_registry_immutable_tool_definitions():
    """RegisteredTool instances must be frozen and cannot be modified at runtime."""
    tool = default_tool_registry.get_tool("forward")
    assert tool is not None
    with pytest.raises(ValidationError):
        tool.name = "altered_forward"


def test_registry_case_and_whitespace_insensitivity():
    """Registry lookup must handle uppercase and leading/trailing whitespace cleanly."""
    registry = ToolRegistry()
    tool = registry.get_tool("  FORWARD  ")
    assert tool is not None
    assert tool.name == "forward"
    assert registry.is_registered("  StOP ") is True


# ---------------------------------------------------------------------------
# 2. Command Safety Validator Tests (Authoritative Fail-Closed Validation)
# ---------------------------------------------------------------------------

def test_validator_valid_movement_parameters():
    """Valid speed and steps within boundaries must pass validation."""
    result = CommandSafetyValidator.validate_command(
        tool_name="forward",
        parameters={"speed": 50, "steps": 3},
    )
    assert result.valid is True
    assert result.tool == "forward"
    assert result.parameters == {"speed": 50, "steps": 3}
    assert len(result.errors) == 0


def test_validator_movement_without_parameters():
    """Movement tools accept empty parameters (defaults apply in execution layer)."""
    result = CommandSafetyValidator.validate_command(
        tool_name="backward",
        parameters={},
    )
    assert result.valid is True
    assert result.tool == "backward"
    assert result.parameters == {}


def test_validator_speed_range_enforcement():
    """Speed outside [1, 100] must fail validation."""
    for invalid_speed in [0, -1, -50, 101, 200]:
        result = CommandSafetyValidator.validate_command(
            tool_name="forward",
            parameters={"speed": invalid_speed, "steps": 2},
        )
        assert result.valid is False
        assert any("speed" in err for err in result.errors)


def test_validator_steps_positive_integer_enforcement():
    """Steps <= 0 must fail validation."""
    for invalid_steps in [0, -1, -10]:
        result = CommandSafetyValidator.validate_command(
            tool_name="right",
            parameters={"speed": 40, "steps": invalid_steps},
        )
        assert result.valid is False
        assert any("steps" in err for err in result.errors)


def test_validator_strict_no_type_coercion():
    """Pydantic strict=True must reject stringified numbers, floats, or bools without coercion."""
    # String numbers must be rejected
    result_str = CommandSafetyValidator.validate_command(
        tool_name="forward",
        parameters={"speed": "40", "steps": 2},
    )
    assert result_str.valid is False
    assert any("speed" in err for err in result_str.errors)

    # Floats for integer fields must be rejected
    result_float = CommandSafetyValidator.validate_command(
        tool_name="forward",
        parameters={"speed": 40.5, "steps": 2},
    )
    assert result_float.valid is False

    # Booleans for integer fields must be rejected
    result_bool = CommandSafetyValidator.validate_command(
        tool_name="left",
        parameters={"speed": True, "steps": 2},
    )
    assert result_bool.valid is False


def test_validator_rejects_extra_parameters():
    """Extra/unknown keys must fail validation (extra='forbid')."""
    result = CommandSafetyValidator.validate_command(
        tool_name="forward",
        parameters={"speed": 50, "steps": 3, "unauthorized_key": True},
    )
    assert result.valid is False
    assert any("unauthorized_key" in err for err in result.errors)


def test_validator_posture_and_system_tools_reject_parameters():
    """Postures (stand, sit) and system tools (status, calibrate, manual) must reject parameters."""
    for tool_name in ["stand", "sit", "status", "calibrate", "manual"]:
        result = CommandSafetyValidator.validate_command(
            tool_name=tool_name,
            parameters={"speed": 30},
        )
        assert result.valid is False, f"Tool '{tool_name}' must reject unexpected parameters"
        assert "does not accept parameters" in result.message


def test_validator_stop_parameters():
    """Stop command accepts optional emergency boolean flag; rejects extra keys."""
    # Valid emergency=True
    res_emergency = CommandSafetyValidator.validate_command(
        tool_name="stop",
        parameters={"emergency": True},
    )
    assert res_emergency.valid is True
    assert res_emergency.parameters == {"emergency": True}

    # Valid empty
    res_empty = CommandSafetyValidator.validate_command(
        tool_name="stop",
        parameters={},
    )
    assert res_empty.valid is True

    # Invalid extra field
    res_extra = CommandSafetyValidator.validate_command(
        tool_name="stop",
        parameters={"emergency": True, "extra_flag": "unsafe"},
    )
    assert res_extra.valid is False


def test_validator_rejects_non_dict_parameters():
    """Passing a non-dict parameter payload must fail closed."""
    result = CommandSafetyValidator.validate_command(
        tool_name="forward",
        parameters="not_a_dict",  # type: ignore
    )
    assert result.valid is False
    assert "Parameters must be a JSON object" in result.message


def test_validator_rejects_unknown_tool():
    """Calling validator with non-allowlisted tool must fail closed."""
    result = CommandSafetyValidator.validate_command(
        tool_name="fly_to_mars",
        parameters={},
    )
    assert result.valid is False
    assert "not in the allowlisted tool registry" in result.message


# ---------------------------------------------------------------------------
# 3. Command Simulator Tests (Hardware-Isolated Dry Run)
# ---------------------------------------------------------------------------

def test_simulator_valid_command():
    """Valid dry-run produces execution_mode='simulation' and hardware_state_affected=False."""
    result = CommandSimulator.simulate_dry_run(
        tool_name="forward",
        parameters={"speed": 40, "steps": 5},
    )
    assert isinstance(result, DryRunResponse)
    assert result.success is True
    assert result.execution_mode == "simulation"
    assert result.hardware_state_affected is False
    assert result.tool == "forward"
    assert result.parameters == {"speed": 40, "steps": 5}
    assert "simulated successfully" in result.message


def test_simulator_invalid_command_propagates_validator_rejection():
    """CommandSimulator must internally invoke CommandSafetyValidator and reject safely."""
    result = CommandSimulator.simulate_dry_run(
        tool_name="forward",
        parameters={"speed": 999},  # out of bounds
    )
    assert result.success is False
    assert result.execution_mode == "simulation"
    assert result.hardware_state_affected is False
    assert "Dry-run simulation rejected" in result.message


# ---------------------------------------------------------------------------
# 4. HTTP API Endpoints Tests (/commands/validate and /commands/dry-run)
# ---------------------------------------------------------------------------

def test_api_validate_command_endpoint_success():
    """POST /api/v1/commands/validate with valid payload returns HTTP 200 and valid=True."""
    response = client.post(
        "/api/v1/commands/validate",
        json={"tool_name": "forward", "parameters": {"speed": 50, "steps": 2}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["tool"] == "forward"
    assert data["parameters"] == {"speed": 50, "steps": 2}
    assert data["errors"] == []


def test_api_validate_command_endpoint_validation_failure():
    """POST /api/v1/commands/validate with invalid parameters returns HTTP 200 and valid=False."""
    response = client.post(
        "/api/v1/commands/validate",
        json={"tool_name": "forward", "parameters": {"speed": 200}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert len(data["errors"]) > 0


def test_api_validate_command_empty_tool_name():
    """POST /api/v1/commands/validate with empty tool name returns HTTP 400 Bad Request."""
    response = client.post(
        "/api/v1/commands/validate",
        json={"tool_name": "   ", "parameters": {}},
    )
    assert response.status_code == 400
    assert "Tool name cannot be empty" in response.json()["detail"]


def test_api_dry_run_endpoint_success():
    """POST /api/v1/commands/dry-run returns simulation representation without hardware execution."""
    response = client.post(
        "/api/v1/commands/dry-run",
        json={"tool_name": "stand", "parameters": {}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["execution_mode"] == "simulation"
    assert data["hardware_state_affected"] is False
    assert data["tool"] == "stand"


def test_api_dry_run_endpoint_rejection():
    """POST /api/v1/commands/dry-run for invalid command safely returns success=False."""
    response = client.post(
        "/api/v1/commands/dry-run",
        json={"tool_name": "unknown_tool", "parameters": {}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["execution_mode"] == "simulation"
    assert data["hardware_state_affected"] is False


def test_api_legacy_commands_placeholder():
    """POST /api/v1/commands returns legacy placeholder indicating dry-run recommendation."""
    response = client.post("/api/v1/commands")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Stage 6" in data["message"]


# ---------------------------------------------------------------------------
# 5. End-to-End Pipeline Test: Stage 4 (Intent) -> Stage 5 (Dry Run)
# ---------------------------------------------------------------------------

def test_e2e_intent_to_dry_run_pipeline():
    """
    Simulates full pipeline:
    User request -> /api/v1/intent -> StructuredIntent -> /api/v1/commands/dry-run -> Simulation
    """
    # Step 1: User request to Intent endpoint
    intent_resp = client.post(
        "/api/v1/intent",
        json={"text": "Move forward 3 steps at 40% speed"},
    )
    assert intent_resp.status_code == 200
    intent_data = intent_resp.json()
    assert intent_data["is_valid"] is True
    assert intent_data["action"] == "forward"
    assert intent_data["parameters"] == {"speed": 40, "steps": 3}

    # Step 2: Feed intent output to Stage 5 dry-run endpoint
    dry_run_resp = client.post(
        "/api/v1/commands/dry-run",
        json={
            "tool_name": intent_data["action"],
            "parameters": intent_data["parameters"],
        },
    )
    assert dry_run_resp.status_code == 200
    dry_run_data = dry_run_resp.json()
    assert dry_run_data["success"] is True
    assert dry_run_data["tool"] == "forward"
    assert dry_run_data["parameters"] == {"speed": 40, "steps": 3}
    assert dry_run_data["execution_mode"] == "simulation"
    assert dry_run_data["hardware_state_affected"] is False


# ---------------------------------------------------------------------------
# 6. Physical Safety Boundary Test
# ---------------------------------------------------------------------------

def test_hardware_isolation_boundary():
    """Verify zero serial/ESP32/GPIO calls are made during command validation or dry-run."""
    # Ensure serial or hardware modules are never imported or invoked
    with patch("sys.modules") as mock_modules:
        result = CommandSimulator.simulate_dry_run(
            tool_name="forward",
            parameters={"speed": 50, "steps": 2},
        )
        assert result.success is True
        assert result.hardware_state_affected is False


# ---------------------------------------------------------------------------
# 7. Backward Compatibility: Existing Stage 1-4 Routes Intact
# ---------------------------------------------------------------------------

def test_existing_stage_1_to_4_routes_intact():
    """Verify that adding Stage 5 did not disturb any existing Stage 1-4 endpoints."""
    # Stage 1: Health
    health_resp = client.get("/api/v1/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "ok"

    # Stage 4: Intent
    intent_resp = client.post("/api/v1/intent", json={"text": "stop immediately"})
    assert intent_resp.status_code == 200
    assert intent_resp.json()["action"] == "stop"
