"""
Integration tests for Stage 6 Physical Execution Layer with Mock ESP32 UDP server.
"""

import asyncio
import socket
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import ControlAuthority, settings
from app.execution.authority import default_authority_manager
from app.execution.transport import UDPTransport
from app.execution.timer import default_execution_controller

client = TestClient(app)


class MockESP32UDPServer:
    """Async UDP server on localhost simulating ESP32 response behavior."""

    def __init__(self, host="127.0.0.1", port=18888, drop_acks=False):
        self.host = host
        self.port = port
        self.drop_acks = drop_acks
        self.received_packets = []
        self.sock = None
        self._running = False
        self._task = None

    async def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.bind((self.host, self.port))
        self._running = True
        self._task = asyncio.create_task(self._serve())

    async def _serve(self):
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                data, addr = await loop.sock_recvfrom(self.sock, 1024)
                packet_str = data.decode("utf-8", errors="ignore").strip()
                self.received_packets.append(packet_str)
                if not self.drop_acks:
                    # Echo back standard MOTOR_ACK
                    await loop.sock_sendto(self.sock, b"MOTOR_ACK\n", addr)
            except asyncio.CancelledError:
                break
            except Exception:
                break

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        if self.sock:
            self.sock.close()


@pytest.fixture(autouse=True)
def reset_authority_and_controller():
    default_authority_manager._authority = ControlAuthority.MANUAL
    yield
    default_authority_manager._authority = ControlAuthority.MANUAL


@pytest.mark.anyio
async def test_ephemeral_port_binding():
    """Verify UDPTransport binds to an OS-assigned ephemeral port (not 8888 or 8889)."""
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 18888))
    server.settimeout(1.0)
    transport = UDPTransport(host="127.0.0.1", port=18888, timeout=0.5)

    loop = asyncio.get_running_loop()

    def _server_respond():
        data, sender_addr = server.recvfrom(1024)
        server.sendto(b"MOTOR_ACK", sender_addr)
        return sender_addr[1]

    try:
        server_future = loop.run_in_executor(None, _server_respond)
        await asyncio.sleep(0.02)
        hw_affected, resp, ack = await transport.send_and_receive("STOP|0|0.00")
        sender_port = await server_future

        assert sender_port != 8888
        assert sender_port != 8889
        assert sender_port > 1024
        assert ack is True
    finally:
        server.close()


def test_execute_endpoint_manual_authority_rejection():
    """When authority is MANUAL, movement commands are rejected with OWNERSHIP_REJECTED."""
    # Ensure authority is MANUAL
    default_authority_manager._authority = ControlAuthority.MANUAL

    response = client.post(
        "/api/v1/commands/execute",
        json={"tool_name": "forward", "parameters": {"speed": 40, "steps": 1}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["status"] == "OWNERSHIP_REJECTED"
    assert data["hardware_state_affected"] is False
    assert data["ack_received"] is False


def test_execute_endpoint_missing_speed_rejected():
    """When speed is missing for locomotion, command is rejected with INVALID_COMMAND."""
    # Authority AI
    default_authority_manager._authority = ControlAuthority.AI

    response = client.post(
        "/api/v1/commands/execute",
        json={"tool_name": "forward", "parameters": {"steps": 1}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["status"] == "INVALID_COMMAND"
    assert "requires an explicit 'speed' parameter" in data["message"]


def test_execute_endpoint_stop_allowed_in_manual():
    """Emergency STOP is permitted even when authority is MANUAL."""
    default_authority_manager._authority = ControlAuthority.MANUAL

    # Mock client stop
    with patch("app.execution.client.ESP32Client.emergency_stop", return_value=(True, True, "MOTOR_ACK")):
        response = client.post(
            "/api/v1/commands/execute",
            json={"tool_name": "stop", "parameters": {}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "EXECUTION_SUCCESS"
        assert data["hardware_state_affected"] is True
        assert data["ack_received"] is True
        assert data["packet_sent"] == "STOP|0|0.00"


def test_execute_endpoint_unverified_pivot_turn_rejected():
    """Left/right commands are rejected when ROBOT_PIVOT_TURNS_VERIFIED is False."""
    default_authority_manager._authority = ControlAuthority.AI

    response = client.post(
        "/api/v1/commands/execute",
        json={"tool_name": "left", "parameters": {"speed": 40}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["status"] == "INVALID_COMMAND"
    assert "unverified on physical hardware and disabled" in data["message"]


def test_execute_endpoint_success_with_mock_esp32():
    """Full execution of forward command with AI authority and simulated ESP32."""
    default_authority_manager._authority = ControlAuthority.AI

    with patch("app.execution.client.ESP32Client.send_motor_packet", return_value=(True, True, "MOTOR_ACK")), \
         patch("app.execution.client.ESP32Client.emergency_stop", return_value=(True, True, "MOTOR_ACK")):

        response = client.post(
            "/api/v1/commands/execute",
            json={"tool_name": "forward", "parameters": {"speed": 40, "steps": 1}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "EXECUTION_SUCCESS"
        assert data["hardware_state_affected"] is True
        assert data["ack_received"] is True
        assert data["packet_sent"] == "FORWARD|40|0.00"
        assert data["duration_seconds"] is not None


def test_execute_endpoint_dropped_ack_timeout():
    """When ESP32 does not return ACK, returns ACK_TIMEOUT with hardware_state_affected=True."""
    default_authority_manager._authority = ControlAuthority.AI

    from app.execution.errors import AckTimeoutError
    with patch("app.execution.client.ESP32Client.send_motor_packet", side_effect=AckTimeoutError("No ACK received")), \
         patch("app.execution.client.ESP32Client.emergency_stop", return_value=(True, True, "MOTOR_ACK")):

        response = client.post(
            "/api/v1/commands/execute",
            json={"tool_name": "forward", "parameters": {"speed": 40, "steps": 1}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["status"] == "ACK_TIMEOUT"
        assert data["hardware_state_affected"] is True
        assert data["ack_received"] is False


def test_execute_endpoint_robot_unreachable():
    """When host cannot be resolved, returns ROBOT_UNREACHABLE with hardware_state_affected=False."""
    default_authority_manager._authority = ControlAuthority.AI

    from app.execution.errors import RobotUnreachableError
    with patch("app.execution.client.ESP32Client.send_motor_packet", side_effect=RobotUnreachableError("DNS error")):

        response = client.post(
            "/api/v1/commands/execute",
            json={"tool_name": "forward", "parameters": {"speed": 40, "steps": 1}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["status"] == "ROBOT_UNREACHABLE"
        assert data["hardware_state_affected"] is False
