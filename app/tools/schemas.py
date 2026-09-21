"""
JSON Schema tool definitions for LLM tool calling in IOFT Humanoid Robot.
Conforms to OpenAI / standard JSON-Schema tool specification.
"""

from typing import List, Dict, Any
from app.tools.definitions import is_tool_enabled

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "forward": {
        "name": "forward",
        "description": "Move the robot forward linearly for a specified duration/step count.",
        "parameters": {
            "type": "object",
            "properties": {
                "speed_profile": {
                    "type": "string",
                    "enum": ["slow", "medium", "fast"],
                    "description": "Semantic speed profile: 'slow' (20%), 'medium' (40%), or 'fast' (70%).",
                },
                "requested_speed": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Explicit integer speed percentage between 1 and 100.",
                },
                "steps": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "Positive step count (each step is 0.5s). Defaults to 1 if omitted.",
                },
            },
            "required": [],
        },
    },
    "backward": {
        "name": "backward",
        "description": "Move the robot backward linearly for a specified duration/step count.",
        "parameters": {
            "type": "object",
            "properties": {
                "speed_profile": {
                    "type": "string",
                    "enum": ["slow", "medium", "fast"],
                    "description": "Semantic speed profile: 'slow' (20%), 'medium' (40%), or 'fast' (70%).",
                },
                "requested_speed": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Explicit integer speed percentage between 1 and 100.",
                },
                "steps": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "Positive step count (each step is 0.5s). Defaults to 1 if omitted.",
                },
            },
            "required": [],
        },
    },
    "left": {
        "name": "left",
        "description": "Pivot turn left on the spot (requires physical verification).",
        "parameters": {
            "type": "object",
            "properties": {
                "speed_profile": {
                    "type": "string",
                    "enum": ["slow", "medium", "fast"],
                    "description": "Semantic speed profile: 'slow' (20%), 'medium' (40%), or 'fast' (70%).",
                },
                "requested_speed": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Explicit integer speed percentage between 1 and 100.",
                },
                "steps": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "Positive step count (each step is 0.5s). Defaults to 1 if omitted.",
                },
            },
            "required": [],
        },
    },
    "right": {
        "name": "right",
        "description": "Pivot turn right on the spot (requires physical verification).",
        "parameters": {
            "type": "object",
            "properties": {
                "speed_profile": {
                    "type": "string",
                    "enum": ["slow", "medium", "fast"],
                    "description": "Semantic speed profile: 'slow' (20%), 'medium' (40%), or 'fast' (70%).",
                },
                "requested_speed": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Explicit integer speed percentage between 1 and 100.",
                },
                "steps": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "Positive step count (each step is 0.5s). Defaults to 1 if omitted.",
                },
            },
            "required": [],
        },
    },
    "stop": {
        "name": "stop",
        "description": "Immediately stop all robot locomotion and brake motors.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


FORWARD_TOOL_SCHEMA = TOOL_SCHEMAS["forward"]
BACKWARD_TOOL_SCHEMA = TOOL_SCHEMAS["backward"]
STOP_TOOL_SCHEMA = TOOL_SCHEMAS["stop"]


def get_available_tool_schemas() -> List[Dict[str, Any]]:
    """Return JSON schemas for only currently enabled tools."""
    return [
        schema for name, schema in TOOL_SCHEMAS.items()
        if is_tool_enabled(name)
    ]
