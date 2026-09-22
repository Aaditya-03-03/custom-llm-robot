"""
Integration Tests for Read-Only GET /api/v1/robot/execution/{execution_id} Endpoint.
Verifies:
- 200 OK and valid ExecutionRecord payload for existing execution.
- 404 Not Found for non-existent execution_id.
- Zero UDP hardware packets transmitted by the endpoint.
- Correct serialization of Stage 9 fields.
"""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.verification.models import ExecutionPhase, VerificationStatus
from app.verification.manager import default_execution_manager

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_execution_manager():
    default_execution_manager.clear()
    yield
    default_execution_manager.clear()


@pytest.mark.anyio
async def test_get_execution_record_success():
    """Verify GET /api/v1/robot/execution/{execution_id} returns 200 with valid schema."""
    # Seed a record
    record = await default_execution_manager.create_record(
        execution_id="exec-12345",
        request_id="req-abc",
        session_id="sess-xyz",
        plan_id="plan-999",
        action_id="act-1",
        command="FORWARD",
        speed=40,
        steps=2,
    )
    await default_execution_manager.advance_phase("exec-12345", ExecutionPhase.NOT_VERIFIED)

    response = client.get("/api/v1/robot/execution/exec-12345")
    assert response.status_code == 200
    data = response.json()

    assert data["execution_id"] == "exec-12345"
    assert data["request_id"] == "req-abc"
    assert data["session_id"] == "sess-xyz"
    assert data["plan_id"] == "plan-999"
    assert data["action_id"] == "act-1"
    assert data["command"] == "FORWARD"
    assert data["speed"] == 40
    assert data["steps"] == 2
    assert data["execution_phase"] == ExecutionPhase.NOT_VERIFIED.value
    assert data["verification_status"] == VerificationStatus.PENDING.value
    assert data["physical_motion_verified"] is None
    assert "started_at" in data
    assert "verification_deadline" in data
    assert data["verification_deadline"] is not None


def test_get_execution_record_not_found():
    """Verify GET /api/v1/robot/execution/{execution_id} returns 404 for unknown execution_id."""
    response = client.get("/api/v1/robot/execution/non_existent_id")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.anyio
async def test_get_execution_endpoint_transmits_zero_udp_packets():
    """Verify GET /api/v1/robot/execution/{execution_id} is strictly read-only."""
    await default_execution_manager.create_record(
        execution_id="exec-zero-udp",
        request_id="req-1",
        session_id="sess-1",
        plan_id="plan-1",
        action_id="act-1",
        command="STOP",
    )

    with patch("app.execution.transport.UDPTransport.send_and_receive") as mock_udp:
        response = client.get("/api/v1/robot/execution/exec-zero-udp")
        assert response.status_code == 200
        assert mock_udp.call_count == 0
