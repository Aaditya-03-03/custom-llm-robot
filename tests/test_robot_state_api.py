"""
Integration Tests for Stage 8 Robot State API & Assistant Stale-State Safety Boundary.
Verifies:
- GET /api/v1/robot/state returns valid RobotState snapshot.
- GET /api/v1/robot/state is strictly read-only (zero UDP packets dispatched).
- Assistant rejects movement commands when state is stale (ASSISTANT_ROBOT_UNAVAILABLE).
- Assistant permits STOP commands even when state is stale (failsafe).
- Assistant audit records capture state_before and state_after snapshots.
- GET /api/v1/health reports robot_connected and robot_state_available.
"""

from unittest.mock import patch, AsyncMock
from datetime import datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import ControlAuthority, settings
from app.execution.authority import default_authority_manager
from app.planning.models import AssistantStatus
from app.state.manager import default_state_manager
from app.state.models import ConnectionStatus
from app.schemas.commands import PhysicalExecutionResponse, ExecutionStatus

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_state_and_authority():
    default_authority_manager._authority = ControlAuthority.AI
    original_provider = settings.LLM_PROVIDER
    settings.LLM_PROVIDER = "mock"
    # Fresh state setup
    default_state_manager.record_heartbeat()
    yield
    default_authority_manager._authority = ControlAuthority.MANUAL
    settings.LLM_PROVIDER = original_provider


def test_get_robot_state_endpoint_success():
    """Verify GET /api/v1/robot/state returns current valid RobotState."""
    response = client.get("/api/v1/robot/state")
    assert response.status_code == 200
    data = response.json()
    assert "connection_status" in data
    assert "authority" in data
    assert "execution_status" in data
    assert "motion_state" in data
    assert "watchdog_status" in data
    assert "is_stale" in data


def test_get_robot_state_endpoint_is_strictly_read_only():
    """Verify GET /api/v1/robot/state never transmits UDP packets to the ESP32."""
    with patch("app.execution.transport.UDPTransport.send_and_receive") as mock_udp:
        response = client.get("/api/v1/robot/state")
        assert response.status_code == 200
        assert mock_udp.call_count == 0


def test_assistant_rejects_movement_when_robot_state_stale():
    """
    CRITICAL SAFETY TEST:
    When robot state is STALE or DISCONNECTED, assistant must reject movement commands
    with ASSISTANT_ROBOT_UNAVAILABLE and send ZERO motor packets.
    """
    # Make state stale
    default_state_manager._last_heartbeat_timestamp = (
        datetime.now(timezone.utc) - timedelta(seconds=10)
    )
    assert default_state_manager.get_state().is_stale is True

    with patch("app.execution.adapter.default_execution_adapter.execute_command", new_callable=AsyncMock) as mock_exec:
        response = client.post("/api/v1/assistant", json={"message": "Move forward slowly"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == AssistantStatus.ASSISTANT_ROBOT_UNAVAILABLE.value
        assert data["executed"] is False
        assert "unavailable or stale" in data["response_text"].lower()
        # Zero motor execution calls allowed!
        assert mock_exec.call_count == 0


def test_assistant_permits_stop_when_robot_state_stale():
    """
    CRITICAL SAFETY TEST:
    Emergency STOP commands MUST remain permitted even when robot state is STALE.
    """
    # Make state stale
    default_state_manager._last_heartbeat_timestamp = (
        datetime.now(timezone.utc) - timedelta(seconds=10)
    )
    assert default_state_manager.get_state().is_stale is True

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
        mock_exec.return_value = resp_stop
        response = client.post("/api/v1/assistant", json={"message": "Stop immediately"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == AssistantStatus.ASSISTANT_EXECUTED.value
        assert data["executed"] is True
        assert mock_exec.call_count == 1


def test_assistant_audit_records_include_state_snapshots():
    """Verify audit logging captures state_before and state_after snapshots."""
    # Ensure fresh state
    default_state_manager.record_heartbeat()

    resp_mock = PhysicalExecutionResponse(
        success=True,
        status=ExecutionStatus.EXECUTION_SUCCESS,
        execution_id="audit_exec",
        tool="forward",
        parameters={"speed": 20, "steps": 1},
        packet_sent="FORWARD|20|0.00",
        ack_received=True,
        hardware_state_affected=True,
        duration_seconds=0.6,
        message="Success",
    )

    with patch("app.execution.adapter.default_execution_adapter.execute_command", new_callable=AsyncMock) as mock_exec, \
         patch("app.api.assistant._log_assistant_audit") as mock_audit:
        mock_exec.return_value = resp_mock
        response = client.post("/api/v1/assistant", json={"message": "Move forward slowly"})
        assert response.status_code == 200

        assert mock_audit.call_count == 1
        _, kwargs = mock_audit.call_args
        assert "state_before" in kwargs
        assert "state_after" in kwargs
        assert kwargs["state_before"] is not None
        assert kwargs["state_after"] is not None


def test_health_endpoint_reports_robot_state_availability():
    """Verify /api/v1/health reports robot_state_available and robot_connected."""
    # Connected case
    default_state_manager.record_heartbeat()
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["robot_state_available"] is True
    assert data["robot_connected"] is True

    # Disconnected case
    default_state_manager.mark_disconnected()
    res_disc = client.get("/api/v1/health")
    assert res_disc.status_code == 200
    data_disc = res_disc.json()
    assert data_disc["robot_state_available"] is True
    assert data_disc["robot_connected"] is False
