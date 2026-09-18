"""
Deterministic natural language normalizer and rule-based intent matcher.
Provides 0ms latency matching for common, unambiguous commands and parameter extraction.
"""

import re
from typing import Optional, Tuple
from app.intent.schemas import (
    IntentCategory,
    RobotAction,
    MovementParameters,
    StopParameters,
    StructuredIntent,
)

# Canonical action keywords & patterns
_MOVEMENT_PATTERNS = {
    RobotAction.FORWARD: [
        r"\b(move\s+forward|go\s+forward|walk\s+forward|step\s+forward|forward|ahead|straight)\b",
    ],
    RobotAction.BACKWARD: [
        r"\b(move\s+backward|go\s+backward|walk\s+backward|step\s+backward|backward|back\s+up|reverse)\b",
    ],
    RobotAction.LEFT: [
        r"\b(move\s+left|go\s+left|walk\s+left|step\s+left|left)\b",
    ],
    RobotAction.RIGHT: [
        r"\b(move\s+right|go\s+right|walk\s+right|step\s+right|right)\b",
    ],
}

_POSTURE_PATTERNS = {
    RobotAction.STAND: [r"\b(stand\s+up|stand|rise|get\s+up)\b"],
    RobotAction.SIT: [r"\b(sit\s+down|sit|rest\s+position|rest)\b"],
}

_SYSTEM_PATTERNS = {
    RobotAction.STATUS: [r"\b(status|check\s+status|report\s+status|battery|health|telemetry)\b"],
    RobotAction.CALIBRATE: [r"\b(calibrate|calibration|zero\s+sensors|zero\s+joints)\b"],
    RobotAction.MANUAL: [r"\b(manual\s+mode|switch\s+to\s+manual|manual\s+control|manual)\b"],
}

_UNSUPPORTED_ACTIONS = [
    r"\b(fly|jump|run|sprint|crawl|crouch|ready|cook|clean|dance|swim)\b",
]

_CONVERSATION_STARTERS = [
    r"^(what|who|where|when|why|how|explain|describe|tell\s+me|is\s+there|are\s+you|can\s+you\s+explain)\b",
]


def _extract_movement_parameters(text: str) -> MovementParameters:
    """Extract speed and steps parameters from natural language text."""
    speed: Optional[int] = None
    steps: Optional[int] = None

    # Speed extraction: e.g. "speed 50", "at 40% speed", "speed: 80"
    speed_match = re.search(r"\b(?:speed\s*(?:of|at|:)?\s*|at\s+)(\d{1,3})\s*(?:%|\s*percent)?\b", text, re.IGNORECASE)
    if speed_match:
        val = int(speed_match.group(1))
        if 1 <= val <= 100:
            speed = val

    # Steps extraction: e.g. "3 steps", "5 step"
    steps_match = re.search(r"\b(\d{1,3})\s+steps?\b", text, re.IGNORECASE)
    if steps_match:
        val = int(steps_match.group(1))
        if val > 0:
            steps = val

    return MovementParameters(speed=speed, steps=steps)


def normalize_deterministic_intent(text: str) -> Optional[StructuredIntent]:
    """
    Attempt to deterministically match a user instruction to an established intent.
    Returns StructuredIntent if a clear match is found, or None if ambiguous.
    """
    clean = text.strip().lower()

    # 1. Stop / E-Stop check (highest priority safety rule)
    if re.search(r"\b(emergency\s+stop|e-stop|estop|halt\s+immediately|stop\s+now)\b", clean):
        return StructuredIntent(
            category=IntentCategory.ROBOT_COMMAND,
            action=RobotAction.STOP,
            parameters=StopParameters(emergency=True),
            raw_input=text,
            is_valid=True,
        )
    if re.search(r"\b(stop|halt|freeze|don'?t\s+move|cease)\b", clean):
        return StructuredIntent(
            category=IntentCategory.ROBOT_COMMAND,
            action=RobotAction.STOP,
            parameters=StopParameters(emergency=False),
            raw_input=text,
            is_valid=True,
        )

    # 2. Unsupported actions check (e.g. fly, jump, cook)
    for pat in _UNSUPPORTED_ACTIONS:
        if re.search(pat, clean):
            return StructuredIntent(
                category=IntentCategory.UNSUPPORTED,
                action=None,
                parameters=None,
                raw_input=text,
                is_valid=False,
                error_message=f"Requested action is unsupported by the robot.",
            )

    # 3. Explicit conversation starter check
    for pat in _CONVERSATION_STARTERS:
        if re.search(pat, clean):
            return StructuredIntent(
                category=IntentCategory.CONVERSATION,
                action=None,
                parameters=None,
                raw_input=text,
                is_valid=True,
            )

    # 4. Postures (STAND, SIT)
    for action, patterns in _POSTURE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, clean):
                return StructuredIntent(
                    category=IntentCategory.ROBOT_COMMAND,
                    action=action,
                    parameters=None,
                    raw_input=text,
                    is_valid=True,
                )

    # 5. System commands (STATUS, CALIBRATE, MANUAL)
    for action, patterns in _SYSTEM_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, clean):
                return StructuredIntent(
                    category=IntentCategory.ROBOT_COMMAND,
                    action=action,
                    parameters=None,
                    raw_input=text,
                    is_valid=True,
                )

    # 6. Movement actions (FORWARD, BACKWARD, LEFT, RIGHT)
    for action, patterns in _MOVEMENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, clean):
                params = _extract_movement_parameters(clean)
                has_params = params.speed is not None or params.steps is not None
                return StructuredIntent(
                    category=IntentCategory.ROBOT_COMMAND,
                    action=action,
                    parameters=params if has_params else None,
                    raw_input=text,
                    is_valid=True,
                )

    # Ambiguous or complex phrasing — defer to LLM extractor
    return None
