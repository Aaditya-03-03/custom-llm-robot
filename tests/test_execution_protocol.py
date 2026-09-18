"""
Unit tests for ESP32 wire protocol encoding and decoding.
"""

import pytest
from unittest.mock import patch
from app.execution.protocol import ESP32Protocol
from app.execution.errors import InvalidPacketError, CapabilityDisabledError


def test_protocol_format_stop():
    """STOP must always format as STOP|0|0.00 regardless of parameters."""
    packet = ESP32Protocol.format_packet("stop")
    assert packet == "STOP|0|0.00"

    packet_with_params = ESP32Protocol.format_packet("stop", {"emergency": True})
    assert packet_with_params == "STOP|0|0.00"


def test_protocol_format_forward_and_backward():
    """Forward and backward commands must format as DIRECTION|SPEED|0.00."""
    packet_fwd = ESP32Protocol.format_packet("forward", {"speed": 40, "steps": 3})
    assert packet_fwd == "FORWARD|40|0.00"

    packet_bwd = ESP32Protocol.format_packet("backward", {"speed": 35})
    assert packet_bwd == "BACKWARD|35|0.00"


def test_protocol_requires_explicit_speed_for_locomotion():
    """Option A: Missing or None speed must be rejected with InvalidPacketError."""
    with pytest.raises(InvalidPacketError) as exc:
        ESP32Protocol.format_packet("forward", {})
    assert "requires an explicit 'speed' parameter" in str(exc.value)

    with pytest.raises(InvalidPacketError):
        ESP32Protocol.format_packet("forward", {"speed": None})


def test_protocol_rejects_out_of_range_or_invalid_speed():
    """Speed outside 1-100 or non-int must be rejected."""
    for bad_speed in [0, -10, 101, 200, "50", 40.5, True]:
        with pytest.raises(InvalidPacketError):
            ESP32Protocol.format_packet("forward", {"speed": bad_speed})


def test_protocol_pivot_turns_disabled_by_default():
    """By default, ROBOT_PIVOT_TURNS_VERIFIED is False, rejecting left/right."""
    with pytest.raises(CapabilityDisabledError) as exc:
        ESP32Protocol.format_packet("left", {"speed": 40})
    assert "unverified on physical hardware and disabled" in str(exc.value)

    with pytest.raises(CapabilityDisabledError):
        ESP32Protocol.format_packet("right", {"speed": 40})


def test_protocol_pivot_turns_allowed_when_verified():
    """When ROBOT_PIVOT_TURNS_VERIFIED is True, left and right format as pivot turns."""
    with patch("app.execution.protocol.settings.ROBOT_PIVOT_TURNS_VERIFIED", True):
        packet_left = ESP32Protocol.format_packet("left", {"speed": 50})
        assert packet_left == "LEFT|50|0.00"

        packet_right = ESP32Protocol.format_packet("right", {"speed": 60})
        assert packet_right == "RIGHT|60|0.00"


def test_protocol_rejects_unmapped_tool():
    """Non-locomotion tools must raise InvalidPacketError."""
    for unmapped in ["stand", "sit", "status", "fly", "jump"]:
        with pytest.raises(InvalidPacketError):
            ESP32Protocol.format_packet(unmapped, {})


def test_protocol_ack_parsing():
    """is_valid_ack checks exact MOTOR_ACK token."""
    assert ESP32Protocol.is_valid_ack(b"MOTOR_ACK") is True
    assert ESP32Protocol.is_valid_ack(b"MOTOR_ACK\r\n") is True
    assert ESP32Protocol.is_valid_ack(b"  MOTOR_ACK  ") is True
    assert ESP32Protocol.is_valid_ack(b"ACK") is False
    assert ESP32Protocol.is_valid_ack(b"ERROR") is False
    assert ESP32Protocol.is_valid_ack(b"") is False
    assert ESP32Protocol.is_valid_ack(None) is False
