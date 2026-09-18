"""
Stage 4 Intent Detection and Structured Command Tests.
Verifies grounded capability taxonomy, parameter validation, fail-safe LLM parsing,
endpoint contracts, and the strict zero-execution safety boundary.
"""

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from app.main import app
from app.core.config import settings
from app.intent.schemas import (
    IntentCategory,
    RobotAction,
    MovementParameters,
    StopParameters,
    StructuredIntent,
)
from app.intent.normalizer import normalize_deterministic_intent
from app.intent.extractor import IntentExtractor

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Deterministic Normalizer & Grounded Capabilities
# ---------------------------------------------------------------------------

def test_deterministic_forward_movement():
    """Verify forward movement without parameters."""
    result = normalize_deterministic_intent("Move forward")
    assert result is not None
    assert result.category == IntentCategory.ROBOT_COMMAND
    assert result.action == RobotAction.FORWARD
    assert result.is_valid is True
    assert result.parameters is None


def test_deterministic_movement_with_parameters():
    """Verify extraction of speed and steps."""
    result = normalize_deterministic_intent("Walk forward at 40% speed with 3 steps")
    assert result is not None
    assert result.category == IntentCategory.ROBOT_COMMAND
    assert result.action == RobotAction.FORWARD
    assert result.parameters is not None
    assert isinstance(result.parameters, MovementParameters)
    assert result.parameters.speed == 40
    assert result.parameters.steps == 3


def test_exact_semantics_left_right():
    """
    Verify exact directional semantics for LEFT and RIGHT.
    Must NOT be converted into rotational turn commands.
    """
    left_result = normalize_deterministic_intent("Move left")
    assert left_result is not None
    assert left_result.action == RobotAction.LEFT
    assert left_result.category == IntentCategory.ROBOT_COMMAND

    right_result = normalize_deterministic_intent("Move right")
    assert right_result is not None
    assert right_result.action == RobotAction.RIGHT
    assert right_result.category == IntentCategory.ROBOT_COMMAND


def test_deterministic_postures():
    """Verify STAND and SIT postures."""
    stand_res = normalize_deterministic_intent("Stand up")
    assert stand_res is not None
    assert stand_res.action == RobotAction.STAND

    sit_res = normalize_deterministic_intent("Sit down")
    assert sit_res is not None
    assert sit_res.action == RobotAction.SIT


def test_deterministic_stop_and_estop():
    """Verify STOP distinguishes standard halt from emergency stop."""
    stop_res = normalize_deterministic_intent("Stop the robot")
    assert stop_res is not None
    assert stop_res.action == RobotAction.STOP
    assert isinstance(stop_res.parameters, StopParameters)
    assert stop_res.parameters.emergency is False

    estop_res = normalize_deterministic_intent("Emergency stop immediately")
    assert estop_res is not None
    assert estop_res.action == RobotAction.STOP
    assert isinstance(estop_res.parameters, StopParameters)
    assert estop_res.parameters.emergency is True


def test_deterministic_system_commands():
    """Verify STATUS, CALIBRATE, and MANUAL (data-only intents)."""
    for phrase, expected_action in [
        ("Check robot status", RobotAction.STATUS),
        ("Calibrate sensors", RobotAction.CALIBRATE),
        ("Switch to manual mode", RobotAction.MANUAL),
    ]:
        res = normalize_deterministic_intent(phrase)
        assert res is not None
        assert res.action == expected_action
        assert res.category == IntentCategory.ROBOT_COMMAND
        assert res.is_valid is True


def test_deterministic_conversation_discrimination():
    """Verify common conversational question starters are marked as CONVERSATION."""
    for question in [
        "What is the Meta Quest headset?",
        "Who created this robot?",
        "Explain how the AI architecture works",
        "How do you process vision?",
    ]:
        res = normalize_deterministic_intent(question)
        assert res is not None
        assert res.category == IntentCategory.CONVERSATION
        assert res.action is None


def test_deterministic_unsupported_action():
    """Verify unsupported physical actions fail safe."""
    for unsupported in ["Make the robot fly", "Jump over the wall", "Cook dinner"]:
        res = normalize_deterministic_intent(unsupported)
        assert res is not None
        assert res.category == IntentCategory.UNSUPPORTED
        assert res.is_valid is False
        assert res.action is None


# ---------------------------------------------------------------------------
# 2. Extractor, Mock Provider & Fail-Safe LLM Output Handling
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_extractor_with_mock_provider(monkeypatch):
    """Verify IntentExtractor operates cleanly with MockLLMProvider."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    extractor = IntentExtractor()

    # Ambiguous input that falls through normalizer to LLM
    result = await extractor.extract_intent("Would you mind stepping forward a little bit?")
    assert result.category == IntentCategory.ROBOT_COMMAND
    assert result.action == RobotAction.FORWARD
    assert result.is_valid is True


def test_extractor_markdown_fences_handling():
    """Verify LLM output wrapped in markdown ```json ... ``` is parsed cleanly."""
    extractor = IntentExtractor()
    markdown_output = """Here is the structured intent:
```json
{
  "category": "robot_command",
  "action": "backward",
  "parameters": {"speed": 30, "steps": 2},
  "error_message": null
}
```
"""
    result = extractor._parse_llm_output(markdown_output, raw_input="walk back")
    assert result.category == IntentCategory.ROBOT_COMMAND
    assert result.action == RobotAction.BACKWARD
    assert result.is_valid is True
    assert isinstance(result.parameters, MovementParameters)
    assert result.parameters.speed == 30
    assert result.parameters.steps == 2


def test_extractor_malformed_json_fails_safe():
    """Verify malformed JSON does NOT raise an exception and returns safe invalid intent."""
    extractor = IntentExtractor()

    # Case A: Corrupt text with no complete JSON block
    corrupt_output_a = "I cannot fulfill this. {category: 'invalid json here..."
    result_a = extractor._parse_llm_output(corrupt_output_a, raw_input="do something")
    assert result_a.category == IntentCategory.UNSUPPORTED
    assert result_a.is_valid is False
    assert result_a.action is None
    assert result_a.error_message is not None

    # Case B: Syntactically invalid JSON inside complete braces
    corrupt_output_b = "Output: {category: invalid_syntax, action: 'forward'}"
    result_b = extractor._parse_llm_output(corrupt_output_b, raw_input="do something")
    assert result_b.category == IntentCategory.UNSUPPORTED
    assert result_b.is_valid is False
    assert result_b.action is None
    assert "Invalid JSON format" in result_b.error_message



def test_parameter_validation_bounds():
    """Verify invalid parameter values are rejected by schema."""
    with pytest.raises(ValidationError):
        MovementParameters(speed=150)  # > 100

    with pytest.raises(ValidationError):
        MovementParameters(steps=-1)   # <= 0


# ---------------------------------------------------------------------------
# 3. Hard Safety Boundary — ZERO Robot Execution
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_zero_robot_execution_boundary():
    """
    CRITICAL SAFETY TEST: Verify that extracting intents NEVER calls
    physical robot tools (move_joint, grab_object, stop_robot).
    """
    with patch("app.tools.robot_tools.move_joint") as mock_move, \
         patch("app.tools.robot_tools.grab_object") as mock_grab, \
         patch("app.tools.robot_tools.stop_robot") as mock_stop:

        extractor = IntentExtractor()
        for cmd in ["Move forward", "Stop the robot", "Emergency stop", "Stand up"]:
            await extractor.extract_intent(cmd)

        # None of the execution tools must ever be called
        assert mock_move.call_count == 0
        assert mock_grab.call_count == 0
        assert mock_stop.call_count == 0


# ---------------------------------------------------------------------------
# 4. HTTP API Endpoints (/api/v1/intent & /api/v1/chat integration)
# ---------------------------------------------------------------------------

def test_post_intent_endpoint_success():
    """Verify POST /api/v1/intent endpoint contract."""
    response = client.post(
        "/api/v1/intent",
        json={"text": "Move forward 5 steps at speed 60"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "robot_command"
    assert data["action"] == "forward"
    assert data["is_valid"] is True
    assert data["parameters"]["speed"] == 60
    assert data["parameters"]["steps"] == 5


def test_post_intent_endpoint_empty_text_returns_400():
    """Verify empty text rejected with 400 Bad Request."""
    response = client.post(
        "/api/v1/intent",
        json={"text": "   "},
    )
    assert response.status_code == 400


def test_post_intent_unsupported_returns_safe_object():
    """Verify unsupported command returns safe object without HTTP error."""
    response = client.post(
        "/api/v1/intent",
        json={"text": "Fly over the building"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "unsupported"
    assert data["is_valid"] is False
    assert data["action"] is None


def test_chat_endpoint_annotates_deterministic_intent(monkeypatch):
    """
    Verify POST /api/v1/chat attaches intent annotation when command is deterministic,
    without requiring an extra LLM call.
    """
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")

    with patch("app.api.chat.is_mongodb_available", return_value=True), \
         patch("app.api.chat.ConversationManager") as MockMgr, \
         patch("app.api.chat.VectorStoreManager") as MockVS:

        mock_mgr_instance = MockMgr.return_value
        mock_mgr_instance.resolve_session.return_value = "intent-test-session"
        mock_mgr_instance.fetch_previous_history.return_value = []

        mock_vs_instance = MockVS.return_value
        mock_vs_instance.query.return_value = []

        response = client.post(
            "/api/v1/chat",
            json={"message": "Stop the robot"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "intent" in data
        assert data["intent"] is not None
        assert data["intent"]["action"] == "stop"
        assert data["intent"]["category"] == "robot_command"


def test_chat_endpoint_conversational_turn_leaves_intent_none(monkeypatch):
    """
    Verify POST /api/v1/chat leaves intent=None for standard questions,
    preventing any secondary LLM overhead.
    """
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")

    with patch("app.api.chat.is_mongodb_available", return_value=True), \
         patch("app.api.chat.ConversationManager") as MockMgr, \
         patch("app.api.chat.VectorStoreManager") as MockVS:

        mock_mgr_instance = MockMgr.return_value
        mock_mgr_instance.resolve_session.return_value = "chat-test-session"
        mock_mgr_instance.fetch_previous_history.return_value = []

        mock_vs_instance = MockVS.return_value
        mock_vs_instance.query.return_value = []

        response = client.post(
            "/api/v1/chat",
            json={"message": "What platform is used for interaction?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "intent" in data
        assert data["intent"] is None
