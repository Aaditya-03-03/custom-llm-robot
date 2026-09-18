"""
Unit and integration tests for AI Server Execution Authority and state transitions.
"""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import ControlAuthority
from app.execution.authority import ExecutionAuthorityManager, default_authority_manager
from app.execution.errors import AuthorityConflictError, AckTimeoutError
from app.execution.client import ESP32Client

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_default_authority():
    """Ensure authority manager is in MANUAL state for isolated tests."""
    default_authority_manager._authority = ControlAuthority.MANUAL
    default_authority_manager._changed_by = "test_reset"
    yield
    default_authority_manager._authority = ControlAuthority.MANUAL


@pytest.mark.anyio
async def test_authority_defaults_to_manual():
    """Initial server authority must strictly be MANUAL."""
    mgr = ExecutionAuthorityManager()
    assert mgr.current_authority == ControlAuthority.MANUAL

    allowed, msg = mgr.is_execution_allowed("forward")
    assert allowed is False
    assert "MANUAL" in msg


@pytest.mark.anyio
async def test_stop_allowed_unconditionally_in_manual():
    """STOP must be allowed even when authority is MANUAL."""
    mgr = ExecutionAuthorityManager()
    allowed, msg = mgr.is_execution_allowed("stop")
    assert allowed is True
    assert "Emergency STOP is permitted" in msg


@pytest.mark.anyio
async def test_transition_to_ai_enables_movement():
    """Switching to AI authority enables AI locomotion execution."""
    mgr = ExecutionAuthorityManager()
    success, msg = await mgr.set_authority(ControlAuthority.AI, changed_by="test")
    assert success is True
    assert mgr.current_authority == ControlAuthority.AI

    allowed, _ = mgr.is_execution_allowed("forward")
    assert allowed is True


@pytest.mark.anyio
async def test_transition_ai_to_manual_confirmed_stop_success():
    """Transition from AI to MANUAL dispatches STOP and commits state when acknowledged."""
    mgr = ExecutionAuthorityManager()
    await mgr.set_authority(ControlAuthority.AI)

    mock_client = AsyncMock(spec=ESP32Client)
    mock_client.emergency_stop.return_value = (True, True, "MOTOR_ACK received")

    success, msg = await mgr.set_authority(
        ControlAuthority.MANUAL,
        changed_by="test",
        esp32_client=mock_client,
    )
    assert success is True
    assert mgr.current_authority == ControlAuthority.MANUAL
    mock_client.emergency_stop.assert_awaited_once()


@pytest.mark.anyio
async def test_transition_ai_to_manual_aborts_and_retains_ai_on_stop_failure():
    """If STOP fails or times out during transition to MANUAL, retain AI state and raise error."""
    mgr = ExecutionAuthorityManager()
    await mgr.set_authority(ControlAuthority.AI)

    mock_client = AsyncMock(spec=ESP32Client)
    mock_client.emergency_stop.side_effect = AckTimeoutError("No ACK received")

    with pytest.raises(AuthorityConflictError) as exc:
        await mgr.set_authority(
            ControlAuthority.MANUAL,
            changed_by="test",
            esp32_client=mock_client,
        )

    assert "Retaining AI authority" in str(exc.value) or "Failed to confirm STOP" in str(exc.value)
    # State must NOT have transitioned to MANUAL
    assert mgr.current_authority == ControlAuthority.AI


def test_api_get_ownership_endpoint():
    """GET /api/v1/commands/ownership returns current authority dictionary."""
    response = client.get("/api/v1/commands/ownership")
    assert response.status_code == 200
    data = response.json()
    assert "current_authority" in data
    assert "changed_by" in data
    assert "changed_at" in data


def test_api_update_ownership_endpoint_success():
    """POST /api/v1/commands/ownership transitions authority."""
    # 1. Switch to AI
    resp = client.post("/api/v1/commands/ownership", json={"authority": "AI"})
    assert resp.status_code == 200
    assert resp.json()["current_authority"] == "AI"

    # 2. Switch to MANUAL (mocking confirmed STOP)
    with patch("app.execution.authority.ESP32Client.emergency_stop", return_value=(True, True, "MOTOR_ACK")):
        resp2 = client.post("/api/v1/commands/ownership", json={"authority": "MANUAL"})
        assert resp2.status_code == 200
        assert resp2.json()["current_authority"] == "MANUAL"


def test_api_update_ownership_rejects_invalid_value():
    """POST /api/v1/commands/ownership rejects invalid authority values."""
    resp = client.post("/api/v1/commands/ownership", json={"authority": "AUTONOMOUS_DRIVE"})
    assert resp.status_code == 400
    assert "Must be 'MANUAL' or 'AI'" in resp.json()["detail"]
