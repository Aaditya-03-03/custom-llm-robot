"""
Unit tests for TimedExecutionController concurrency, sequential pulses, and STOP semantics.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock
from app.execution.timer import TimedExecutionController
from app.execution.client import ESP32Client
from app.execution.errors import ExecutionBusyError, AckTimeoutError


@pytest.mark.anyio
async def test_single_locomotion_concurrency_busy_rejection():
    """While one locomotion is running, a second locomotion must be rejected with ExecutionBusyError."""
    mock_client = AsyncMock(spec=ESP32Client)
    mock_client.send_motor_packet.return_value = (True, True, "MOTOR_ACK")
    mock_client.emergency_stop.return_value = (True, True, "MOTOR_ACK")

    controller = TimedExecutionController(client=mock_client)

    # Launch task 1 (duration = 2 steps * 0.5s = 1.0s)
    task1 = asyncio.create_task(controller.execute_locomotion("FORWARD|40|0.00", steps=2))
    await asyncio.sleep(0.05)  # Let it start

    assert controller.is_busy is True

    # Try launching task 2
    with pytest.raises(ExecutionBusyError):
        await controller.execute_locomotion("BACKWARD|30|0.00", steps=1)

    # Cancel task1 to clean up
    await controller.execute_stop()
    try:
        await task1
    except asyncio.CancelledError:
        pass


@pytest.mark.anyio
async def test_mid_flight_stop_cancels_active_locomotion():
    """An incoming execute_stop() cancels active locomotion and dispatches STOP|0|0.00."""
    mock_client = AsyncMock(spec=ESP32Client)
    mock_client.send_motor_packet.return_value = (True, True, "MOTOR_ACK")
    mock_client.emergency_stop.return_value = (True, True, "MOTOR_ACK")

    controller = TimedExecutionController(client=mock_client)

    # Start long locomotion
    task = asyncio.create_task(controller.execute_locomotion("FORWARD|40|0.00", steps=4))
    await asyncio.sleep(0.05)
    assert controller.is_busy is True

    # Trigger emergency STOP
    hw_affected, ack_recv, msg = await controller.execute_stop()
    assert hw_affected is True
    assert ack_recv is True

    # Controller must now be idle
    assert controller.is_busy is False
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.anyio
async def test_locomotion_completes_with_explicit_stop():
    """On clean completion, TimedExecutionController sends an explicit STOP packet."""
    mock_client = AsyncMock(spec=ESP32Client)
    mock_client.send_motor_packet.return_value = (True, True, "MOTOR_ACK")
    mock_client.emergency_stop.return_value = (True, True, "MOTOR_ACK")

    controller = TimedExecutionController(client=mock_client)

    # Execute 1 step = 0.5s
    hw_affected, ack_recv, duration, msg = await controller.execute_locomotion("FORWARD|40|0.00", steps=1)

    assert hw_affected is True
    assert ack_recv is True
    assert duration >= 0.5
    # emergency_stop was called for clean termination
    mock_client.emergency_stop.assert_awaited_once()


@pytest.mark.anyio
async def test_pulse_ack_timeout_halts_execution_and_dispatches_stop():
    """If a refresh pulse times out, execution halts and dispatches failsafe STOP."""
    mock_client = AsyncMock(spec=ESP32Client)
    # First pulse succeeds, second pulse times out
    mock_client.send_motor_packet.side_effect = [
        (True, True, "MOTOR_ACK"),
        AckTimeoutError("Timeout on pulse 2"),
    ]
    mock_client.emergency_stop.return_value = (True, True, "MOTOR_ACK")

    controller = TimedExecutionController(client=mock_client)

    with pytest.raises(AckTimeoutError) as exc:
        await controller.execute_locomotion("FORWARD|40|0.00", steps=2)

    assert "Emergency STOP dispatched" in str(exc.value)
    mock_client.emergency_stop.assert_awaited()
