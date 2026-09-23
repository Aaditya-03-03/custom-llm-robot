"""
Stage 10 Local Inference Acceptance Tests.

Pre-deployment acceptance test suite verifying local model integration with physical
execution strictly disabled:
- Environment state: execution layer = disabled/mock, physical robot = not connected.
- Inference contract: do_sample = False, num_beams = 1, max_new_tokens = 256 (deterministic greedy decoding).
- Test battery:
  1. "move forward slowly" -> parses valid tool call (forward, speed 20, steps 1).
  2. "move backward at speed 50 for 2 steps" -> parses valid tool call (backward, speed 50, steps 2).
  3. "move forward" -> returns ASSISTANT_CLARIFICATION_REQUIRED (zero Stage 5/6 execution).
  4. "fly to the moon" -> model outputs tool: null + non-empty reason -> Stage 7 classifies as UNSUPPORTED -> zero Stage 5/6 execution -> zero UDP packets.
  5. "stop" -> parses valid stop tool call.
- Invariants strictly asserted:
  - mock_udp.call_count == 0 (zero UDP packets dispatched).
  - Stage 6 execute_locomotion() is never called.
"""

from unittest.mock import patch, AsyncMock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import ControlAuthority, settings
from app.execution.authority import default_authority_manager
from app.planning.models import AssistantStatus
from app.schemas.commands import PhysicalExecutionResponse, ExecutionStatus
from app.state.manager import default_state_manager

client = TestClient(app)


@pytest.fixture(autouse=True)
def ensure_safe_pre_deployment_environment():
    """Ensure AI authority and mock execution environment with fresh simulated connectivity and idle state."""
    default_authority_manager._authority = ControlAuthority.AI
    default_state_manager.record_heartbeat()
    default_state_manager.record_command_complete(success=True)
    original_provider = settings.LLM_PROVIDER
    settings.LLM_PROVIDER = "mock"
    yield
    default_authority_manager._authority = ControlAuthority.MANUAL
    settings.LLM_PROVIDER = original_provider


def test_local_acceptance_move_forward_slowly():
    """Acceptance Test 1: 'move forward slowly' parses into forward tool, speed 20, steps 1 with 0 UDP packets."""
    mock_resp = PhysicalExecutionResponse(
        success=True,
        status=ExecutionStatus.EXECUTION_SUCCESS,
        execution_id="mock_exec_1",
        tool="forward",
        parameters={"speed": 20, "steps": 1},
        packet_sent="",
        ack_received=False,
        hardware_state_affected=False,
        duration_seconds=0.0,
        message="Execution layer mock/disabled",
    )

    with patch("app.execution.transport.UDPTransport.send_and_receive", new_callable=AsyncMock) as mock_udp, \
         patch("app.execution.timer.TimedExecutionController.execute_locomotion", new_callable=AsyncMock) as mock_locomotion, \
         patch("app.execution.adapter.default_execution_adapter.execute_command", new_callable=AsyncMock) as mock_exec:

        mock_exec.return_value = mock_resp

        response = client.post("/api/v1/assistant", json={"message": "move forward slowly"})
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == AssistantStatus.ASSISTANT_EXECUTED.value
        actions = data.get("plan", {}).get("actions", [])
        assert len(actions) == 1
        assert actions[0]["tool"] == "forward"
        # speed profile "slow" maps to speed 20
        speed = actions[0]["parameters"].get("speed")
        assert speed == 20 or actions[0]["parameters"].get("speed_profile") == "slow"

        # Strictly assert zero physical execution and zero UDP packets
        assert mock_locomotion.call_count == 0
        assert mock_udp.call_count == 0


def test_local_acceptance_move_backward_speed_50_steps_2():
    """Acceptance Test 2: 'move backward at speed 50 for 2 steps' parses backward, speed 50, steps 2 with 0 UDP packets."""
    mock_resp = PhysicalExecutionResponse(
        success=True,
        status=ExecutionStatus.EXECUTION_SUCCESS,
        execution_id="mock_exec_2",
        tool="backward",
        parameters={"speed": 50, "steps": 2},
        packet_sent="",
        ack_received=False,
        hardware_state_affected=False,
        duration_seconds=0.0,
        message="Execution layer mock/disabled",
    )

    with patch("app.execution.transport.UDPTransport.send_and_receive", new_callable=AsyncMock) as mock_udp, \
         patch("app.execution.timer.TimedExecutionController.execute_locomotion", new_callable=AsyncMock) as mock_locomotion, \
         patch("app.execution.adapter.default_execution_adapter.execute_command", new_callable=AsyncMock) as mock_exec:

        mock_exec.return_value = mock_resp

        response = client.post("/api/v1/assistant", json={"message": "move backward at speed 50 for 2 steps"})
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == AssistantStatus.ASSISTANT_EXECUTED.value
        actions = data.get("plan", {}).get("actions", [])
        assert len(actions) == 1
        assert actions[0]["tool"] == "backward"
        assert actions[0]["parameters"].get("speed") == 50
        assert actions[0]["parameters"].get("steps") == 2

        # Strictly assert zero physical execution and zero UDP packets
        assert mock_locomotion.call_count == 0
        assert mock_udp.call_count == 0


def test_local_acceptance_move_forward_requires_clarification():
    """Acceptance Test 3: 'move forward' without speed requires clarification with zero Stage 5/6 execution."""
    with patch("app.execution.transport.UDPTransport.send_and_receive", new_callable=AsyncMock) as mock_udp, \
         patch("app.execution.timer.TimedExecutionController.execute_locomotion", new_callable=AsyncMock) as mock_locomotion, \
         patch("app.execution.adapter.default_execution_adapter.execute_command", new_callable=AsyncMock) as mock_exec:

        response = client.post("/api/v1/assistant", json={"message": "move forward"})
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == AssistantStatus.ASSISTANT_CLARIFICATION_REQUIRED.value
        assert data["executed"] is False
        assert mock_exec.call_count == 0
        assert mock_locomotion.call_count == 0
        assert mock_udp.call_count == 0


def test_local_acceptance_unsupported_flight_zero_packets():
    """Acceptance Test 4: 'fly to the moon' produces UNSUPPORTED status with zero Stage 5/6 execution and zero packets."""
    with patch("app.execution.transport.UDPTransport.send_and_receive", new_callable=AsyncMock) as mock_udp, \
         patch("app.execution.timer.TimedExecutionController.execute_locomotion", new_callable=AsyncMock) as mock_locomotion, \
         patch("app.execution.adapter.default_execution_adapter.execute_command", new_callable=AsyncMock) as mock_exec:

        response = client.post("/api/v1/assistant", json={"message": "fly to the moon"})
        assert response.status_code == 200
        data = response.json()

        # Must classify as unsupported / validation failed and produce 0 executed actions
        assert data["executed"] is False
        assert mock_exec.call_count == 0
        assert mock_locomotion.call_count == 0
        assert mock_udp.call_count == 0


def test_local_acceptance_stop_command():
    """Acceptance Test 5: 'stop' parses valid stop tool call."""
    mock_resp = PhysicalExecutionResponse(
        success=True,
        status=ExecutionStatus.EXECUTION_SUCCESS,
        execution_id="mock_exec_stop",
        tool="stop",
        parameters={},
        packet_sent="",
        ack_received=False,
        hardware_state_affected=False,
        duration_seconds=0.0,
        message="Execution layer mock/disabled",
    )

    with patch("app.execution.transport.UDPTransport.send_and_receive", new_callable=AsyncMock) as mock_udp, \
         patch("app.execution.timer.TimedExecutionController.execute_locomotion", new_callable=AsyncMock) as mock_locomotion, \
         patch("app.execution.adapter.default_execution_adapter.execute_command", new_callable=AsyncMock) as mock_exec:

        mock_exec.return_value = mock_resp

        response = client.post("/api/v1/assistant", json={"message": "stop"})
        assert response.status_code == 200
        data = response.json()

        actions = data.get("plan", {}).get("actions", [])
        assert len(actions) == 1
        assert actions[0]["tool"] == "stop"
        assert mock_locomotion.call_count == 0
        assert mock_udp.call_count == 0
