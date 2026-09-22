"""
Unit & Concurrency Tests for RobotStateManager.
Verifies:
- Command state vs physical state separation (ACK != physical movement).
- Heartbeat PONG semantics (connectivity updated, motion untouched).
- Freshness threshold and transition to STALE/DISCONNECTED.
- Decoupled mark_disconnected() (watchdog remains UNKNOWN).
- Socket Isolation: Concurrent movement and heartbeat transactions without cross-consumption.
"""

import asyncio
import socket
from datetime import datetime, timezone, timedelta
import pytest

from app.state.manager import RobotStateManager
from app.state.models import ConnectionStatus, MotionState, ExecutionState, WatchdogStatus
from app.execution.transport import UDPTransport


def test_command_vs_physical_state_separation():
    """Verify ACK != physical movement invariant."""
    manager = RobotStateManager()
    
    # 1. Start command
    manager.record_command_start("forward", speed=20)
    state_start = manager.get_state()
    assert state_start.requested_command == "FORWARD"
    assert state_start.requested_speed == 20
    assert state_start.current_command == "FORWARD"
    assert state_start.execution_status == ExecutionState.EXECUTING
    assert state_start.motion_state == MotionState.UNKNOWN

    # 2. Receive MOTOR_ACK
    manager.record_ack("MOTOR_ACK")
    state_ack = manager.get_state()
    assert state_ack.last_ack == "MOTOR_ACK"
    assert state_ack.connection_status == ConnectionStatus.CONNECTED
    # Critical invariant: motion state is STILL UNKNOWN!
    assert state_ack.motion_state == MotionState.UNKNOWN

    # 3. Complete execution with STOP
    manager.record_command_complete(success=True)
    state_done = manager.get_state()
    assert state_done.execution_status == ExecutionState.IDLE
    assert state_done.current_command == "STOP"
    assert state_done.last_command == "FORWARD"
    assert state_done.motion_state == MotionState.STOPPED


def test_pong_updates_connectivity_not_motion():
    """Verify PONG confirms reachability without mutating motion state."""
    manager = RobotStateManager()
    assert manager.get_state().connection_status == ConnectionStatus.UNKNOWN

    manager.record_heartbeat()
    state = manager.get_state()
    assert state.connection_status == ConnectionStatus.CONNECTED
    assert state.last_heartbeat_timestamp is not None
    # Motion state must remain UNKNOWN
    assert state.motion_state == MotionState.UNKNOWN


def test_freshness_threshold():
    """Verify state transitions to is_stale=True and DISCONNECTED after 5s."""
    manager = RobotStateManager()
    # Artificially set heartbeat to 6 seconds ago
    past_time = datetime.now(timezone.utc) - timedelta(seconds=6)
    manager._last_heartbeat_timestamp = past_time

    state = manager.get_state()
    assert state.is_stale is True
    assert state.connection_status == ConnectionStatus.DISCONNECTED


def test_mark_disconnected_decoupled_from_watchdog():
    """Verify mark_disconnected sets DISCONNECTED while watchdog remains UNKNOWN."""
    manager = RobotStateManager()
    manager.record_heartbeat()
    assert manager.get_state().connection_status == ConnectionStatus.CONNECTED

    manager.mark_disconnected()
    state = manager.get_state()
    assert state.connection_status == ConnectionStatus.DISCONNECTED
    # Watchdog status must NOT be forced to EXPIRED
    assert state.watchdog_status == WatchdogStatus.UNKNOWN


@pytest.mark.anyio
async def test_socket_isolation_concurrent_movement_and_heartbeat():
    """
    MANDATORY ISOLATION TEST:
    Concurrent movement transaction and heartbeat transaction must run on
    independent ephemeral sockets and receive their own responses without cross-consumption.
    """
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_sock.bind(("127.0.0.1", 18889))
    server_sock.settimeout(2.0)

    received_requests = []
    loop = asyncio.get_running_loop()

    # Echo server responding appropriately to FORWARD vs PING in executor thread
    def _serve():
        for _ in range(2):
            data, addr = server_sock.recvfrom(1024)
            msg = data.decode("utf-8", errors="ignore").strip()
            received_requests.append((msg, addr))
            if "FORWARD" in msg:
                server_sock.sendto(b"MOTOR_ACK\n", addr)
            elif "PING" in msg:
                server_sock.sendto(b"PONG\n", addr)

    server_task = loop.run_in_executor(None, _serve)

    # Two separate UDP transports bound to distinct ephemeral client sockets
    movement_transport = UDPTransport(host="127.0.0.1", port=18889, timeout=1.0)
    heartbeat_transport = UDPTransport(host="127.0.0.1", port=18889, timeout=1.0)

    # Dispatch concurrently
    async def _do_movement():
        _, resp, ack = await movement_transport.send_and_receive("FORWARD|20|0.00\n")
        return resp.strip()

    async def _do_heartbeat():
        _, resp, _ = await heartbeat_transport.send_and_receive("PING\n")
        return resp.strip()

    mv_resp, hb_resp = await asyncio.gather(_do_movement(), _do_heartbeat())
    await server_task
    server_sock.close()

    # Verify no cross-consumption!
    assert mv_resp == b"MOTOR_ACK"
    assert hb_resp == b"PONG"

    # Verify each came from a different ephemeral client port
    assert len(received_requests) == 2
    port_1 = received_requests[0][1][1]
    port_2 = received_requests[1][1][1]
    assert port_1 != port_2, "Each transaction must originate from a distinct ephemeral client port"
