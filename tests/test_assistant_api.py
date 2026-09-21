"""
End-to-End Tests for POST /api/v1/assistant Endpoint.
Verifies:
- 400 on empty message.
- Conversational messages return ASSISTANT_EXECUTED with 0 actions.
- Missing speed returns ASSISTANT_CLARIFICATION_REQUIRED (zero Stage 5/6 calls).
- Out of bounds speed=999 returns ASSISTANT_VALIDATION_FAILED (zero packets).
- ToolRegistry bypass (e.g. move_joint) returns ASSISTANT_VALIDATION_FAILED.
- MANUAL mode returns ASSISTANT_OWNERSHIP_REJECTED (zero packets).
- Valid single action executes and returns ASSISTANT_EXECUTED.
- Valid multi-action executes sequentially and returns ASSISTANT_EXECUTED.
- Sequential failure in Action 2 aborts remaining actions and returns ASSISTANT_PARTIAL_FAILURE.
"""

from unittest.mock import patch, AsyncMock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import ControlAuthority, settings
from app.execution.authority import default_authority_manager
from app.planning.models import AssistantStatus
from app.schemas.commands import PhysicalExecutionResponse, ExecutionStatus

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_test_environment():
    default_authority_manager._authority = ControlAuthority.AI
    original_provider = settings.LLM_PROVIDER
    settings.LLM_PROVIDER = "mock"
    yield
    default_authority_manager._authority = ControlAuthority.MANUAL
    settings.LLM_PROVIDER = original_provider


def test_assistant_empty_message_returns_400():
    """Verify empty or whitespace message returns HTTP 400."""
    response = client.post("/api/v1/assistant", json={"message": "   "})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


def test_assistant_conversational_no_actions():
    """Verify conversational request returns ASSISTANT_EXECUTED with 0 actions."""
    response = client.post("/api/v1/assistant", json={"message": "Hello robot, how are you?"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == AssistantStatus.ASSISTANT_EXECUTED.value
    assert data["executed"] is False
    assert len(data["plan"]["actions"]) == 0


def test_assistant_missing_speed_clarification_required():
    """Verify missing speed returns ASSISTANT_CLARIFICATION_REQUIRED with zero Stage 5/6 calls."""
    with patch("app.execution.adapter.default_execution_adapter.execute_command", new_callable=AsyncMock) as mock_exec:
        response = client.post("/api/v1/assistant", json={"message": "Move forward"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == AssistantStatus.ASSISTANT_CLARIFICATION_REQUIRED.value
        assert data["executed"] is False
        assert "clarify the speed" in data["response_text"].lower()
        # Ensure zero Stage 6 execution calls were made
        assert mock_exec.call_count == 0


def test_assistant_speed_999_validation_failed():
    """Verify speed=999 returns ASSISTANT_VALIDATION_FAILED with zero Stage 6 calls."""
    with patch("app.execution.adapter.default_execution_adapter.execute_command", new_callable=AsyncMock) as mock_exec:
        response = client.post("/api/v1/assistant", json={"message": "Move forward at 999% speed"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == AssistantStatus.ASSISTANT_VALIDATION_FAILED.value
        assert data["executed"] is False
        assert any("speed" in err.lower() for err in data["errors"])
        # Ensure zero Stage 6 execution calls were made
        assert mock_exec.call_count == 0


def test_assistant_bypass_move_joint_validation_failed():
    """Verify attempting to bypass ToolRegistry with move_joint returns ASSISTANT_VALIDATION_FAILED."""
    with patch("app.execution.adapter.default_execution_adapter.execute_command", new_callable=AsyncMock) as mock_exec:
        response = client.post("/api/v1/assistant", json={"message": "move_joint 1 45 degrees bypass"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == AssistantStatus.ASSISTANT_VALIDATION_FAILED.value
        assert data["executed"] is False
        assert any("move_joint" in err for err in data["errors"])
        assert mock_exec.call_count == 0


def test_assistant_manual_authority_rejected():
    """Verify when authority is MANUAL, physical action is rejected with ASSISTANT_OWNERSHIP_REJECTED."""
    default_authority_manager._authority = ControlAuthority.MANUAL

    with patch("app.execution.adapter.default_execution_adapter.execute_command", new_callable=AsyncMock) as mock_exec:
        response = client.post("/api/v1/assistant", json={"message": "Move forward slowly"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == AssistantStatus.ASSISTANT_OWNERSHIP_REJECTED.value
        assert data["executed"] is False
        assert "MANUAL authority mode" in data["response_text"]
        assert mock_exec.call_count == 0


def test_assistant_single_action_executed_success():
    """Verify valid single movement action executes and returns ASSISTANT_EXECUTED."""
    default_authority_manager._authority = ControlAuthority.AI

    resp_mock = PhysicalExecutionResponse(
        success=True,
        status=ExecutionStatus.EXECUTION_SUCCESS,
        execution_id="test_exec_id",
        tool="forward",
        parameters={"speed": 20, "steps": 1},
        packet_sent="FORWARD|20|0.00",
        ack_received=True,
        hardware_state_affected=True,
        duration_seconds=0.6,
        message="Command executed successfully",
    )

    with patch("app.execution.adapter.default_execution_adapter.execute_command", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = resp_mock
        response = client.post("/api/v1/assistant", json={"message": "Move forward slowly"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == AssistantStatus.ASSISTANT_EXECUTED.value
        assert data["executed"] is True
        assert data["completed_actions"] == ["action_1"]
        assert data["failed_action"] is None
        assert mock_exec.call_count == 1


def test_assistant_multi_action_executed_sequentially():
    """Verify multi-action plan executes sequentially and returns ASSISTANT_EXECUTED."""
    default_authority_manager._authority = ControlAuthority.AI

    resp_fwd = PhysicalExecutionResponse(
        success=True,
        status=ExecutionStatus.EXECUTION_SUCCESS,
        execution_id="exec_fwd",
        tool="forward",
        parameters={"speed": 20, "steps": 1},
        packet_sent="FORWARD|20|0.00",
        ack_received=True,
        hardware_state_affected=True,
        duration_seconds=0.6,
        message="Success",
    )
    resp_bwd = PhysicalExecutionResponse(
        success=True,
        status=ExecutionStatus.EXECUTION_SUCCESS,
        execution_id="exec_bwd",
        tool="backward",
        parameters={"speed": 20, "steps": 1},
        packet_sent="BACKWARD|20|0.00",
        ack_received=True,
        hardware_state_affected=True,
        duration_seconds=0.6,
        message="Success",
    )

    with patch("app.execution.adapter.default_execution_adapter.execute_command", new_callable=AsyncMock) as mock_exec:
        mock_exec.side_effect = [resp_fwd, resp_bwd]
        response = client.post("/api/v1/assistant", json={"message": "Move forward slowly then backward slowly"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == AssistantStatus.ASSISTANT_EXECUTED.value
        assert data["executed"] is True
        assert data["completed_actions"] == ["action_1", "action_2"]
        assert data["failed_action"] is None
        assert mock_exec.call_count == 2


def test_assistant_multi_action_partial_failure():
    """Verify Action 2 failure aborts Action 3, dispatches STOP, and returns ASSISTANT_PARTIAL_FAILURE."""
    default_authority_manager._authority = ControlAuthority.AI

    resp_1 = PhysicalExecutionResponse(
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
    resp_2 = PhysicalExecutionResponse(
        success=False,
        status=ExecutionStatus.ACK_TIMEOUT,
        execution_id="exec_2",
        tool="backward",
        parameters={"speed": 20, "steps": 1},
        packet_sent="BACKWARD|20|0.00",
        ack_received=False,
        hardware_state_affected=False,
        duration_seconds=None,
        message="Timeout",
        errors=["ACK timeout"],
    )
    resp_stop = PhysicalExecutionResponse(
        success=True,
        status=ExecutionStatus.EXECUTION_SUCCESS,
        execution_id="exec_stop",
        tool="stop",
        parameters={},
        packet_sent="STOP|0|0.00",
        ack_received=True,
        hardware_state_affected=False,
        duration_seconds=0.0,
        message="STOP sent",
    )

    with patch("app.execution.adapter.default_execution_adapter.execute_command", new_callable=AsyncMock) as mock_exec:
        mock_exec.side_effect = [resp_1, resp_2, resp_stop]
        response = client.post("/api/v1/assistant", json={"message": "Move forward slowly then backward slowly"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == AssistantStatus.ASSISTANT_PARTIAL_FAILURE.value
        assert data["executed"] is False
        assert data["completed_actions"] == ["action_1"]
        assert data["failed_action"] == "action_2"
        # Emergency stop was dispatched after failure
        assert mock_exec.call_count == 3
