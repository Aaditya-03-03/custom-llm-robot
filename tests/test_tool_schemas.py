"""
Tests for Stage 7 Tool Schemas and Runtime Enablement.
Verifies:
- Enabled vs disabled tools in Stage 7 Phase 1.
- Canonical tool schemas match JSON Schema standard.
- Parameter requirements and type definitions.
"""

import pytest
from app.tools.definitions import is_tool_enabled, get_enabled_tools, ENABLED_STAGE_7_TOOLS, DISABLED_STAGE_7_TOOLS
from app.tools.schemas import get_available_tool_schemas, FORWARD_TOOL_SCHEMA, BACKWARD_TOOL_SCHEMA, STOP_TOOL_SCHEMA


def test_stage_7_enabled_tools():
    """Verify only forward, backward, stop are enabled in Phase 1."""
    enabled = get_enabled_tools()
    assert "forward" in enabled
    assert "backward" in enabled
    assert "stop" in enabled
    assert len(enabled) == 3


def test_stage_7_disabled_tools():
    """Verify other grounded capabilities remain disabled for LLM execution."""
    disabled = ["left", "right", "stand", "sit", "status", "calibrate", "manual"]
    for tool_name in disabled:
        assert is_tool_enabled(tool_name) is False
        assert tool_name in DISABLED_STAGE_7_TOOLS


def test_tool_schemas_structure():
    """Verify get_available_tool_schemas provides valid JSON Schema definitions."""
    schemas = get_available_tool_schemas()
    assert len(schemas) == 3
    tool_names = [s["name"] for s in schemas]
    assert set(tool_names) == {"forward", "backward", "stop"}

    # Check forward schema details
    fwd_schema = next(s for s in schemas if s["name"] == "forward")
    assert fwd_schema["parameters"]["type"] == "object"
    props = fwd_schema["parameters"]["properties"]
    assert "speed_profile" in props
    assert props["speed_profile"]["enum"] == ["slow", "medium", "fast"]
    assert "steps" in props
    assert props["steps"]["default"] == 1


def test_stop_tool_schema():
    """Verify stop tool schema does not require speed or steps."""
    stop_schema = STOP_TOOL_SCHEMA
    assert stop_schema["name"] == "stop"
    assert stop_schema["parameters"]["type"] == "object"
    assert "speed_profile" not in stop_schema["parameters"].get("properties", {})
