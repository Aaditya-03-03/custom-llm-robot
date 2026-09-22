"""
Unit tests for Stage 8 TelemetryParser.
Verifies parsing of MOTOR_ACK, PONG, pipe-delimited STATUS, JSON telemetry,
and fail-closed behavior on corrupted/malformed packets.
"""

import json
import pytest

from app.state.telemetry import TelemetryParser, TelemetryType
from app.state.models import ConnectionStatus, MotionState


def test_parse_motor_ack():
    """Verify standard in-band MOTOR_ACK parsing."""
    parsed = TelemetryParser.parse(b"MOTOR_ACK\n")
    assert parsed.packet_type == TelemetryType.MOTOR_ACK
    assert parsed.connection == ConnectionStatus.CONNECTED
    assert parsed.error is None


def test_parse_pong_heartbeat():
    """Verify reachability heartbeat PONG parsing."""
    parsed = TelemetryParser.parse("PONG")
    assert parsed.packet_type == TelemetryType.PONG
    assert parsed.connection == ConnectionStatus.CONNECTED
    assert parsed.error is None


def test_parse_pipe_delimited_status():
    """Verify STATUS|<conn>|<motion>|<command>|<speed> parsing."""
    payload = "STATUS|CONNECTED|MOVING|FORWARD|20"
    parsed = TelemetryParser.parse(payload)
    assert parsed.packet_type == TelemetryType.STATUS
    assert parsed.connection == ConnectionStatus.CONNECTED
    assert parsed.motion == MotionState.MOVING
    assert parsed.command == "FORWARD"
    assert parsed.speed == 20
    assert parsed.error is None


def test_parse_json_status():
    """Verify JSON status telemetry parsing."""
    payload = json.dumps({
        "type": "status",
        "connection": "connected",
        "motion": "moving",
        "command": "forward",
        "speed": 40
    })
    parsed = TelemetryParser.parse(payload)
    assert parsed.packet_type == TelemetryType.STATUS
    assert parsed.connection == ConnectionStatus.CONNECTED
    assert parsed.motion == MotionState.MOVING
    assert parsed.command == "FORWARD"
    assert parsed.speed == 40


def test_reject_malformed_pipe_status():
    """Verify incomplete pipe-delimited status is safely rejected."""
    parsed = TelemetryParser.parse("STATUS|CONNECTED|MOVING")
    assert parsed.packet_type == TelemetryType.UNKNOWN
    assert parsed.error is not None
    assert "expected 5 fields" in parsed.error


def test_reject_malformed_json():
    """Verify invalid JSON is safely rejected without raising exceptions."""
    parsed = TelemetryParser.parse("{broken json...")
    assert parsed.packet_type == TelemetryType.UNKNOWN
    assert parsed.error is not None
