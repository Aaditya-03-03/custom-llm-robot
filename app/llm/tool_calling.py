"""
LLM Prompting & Tool Calling Assembly for Qwen 2.5 3B Action Planning.
Formats tool schemas and few-shot examples for structured robot tool proposals.
"""

import json
from typing import List, Dict, Any, Optional

from app.schemas.chat import ChatMessage
from app.tools.schemas import get_available_tool_schemas
from app.core.config import settings


TOOL_CALLING_SYSTEM_PROMPT = """\
You are the Robot Action Planning Engine for the IOFT Robot.
Your task is to translate user natural language requests into structured tool calls.

Available Tools:
{tool_schemas}

RULES:
1. You may propose actions using ONLY the available tools above. Do not invent any tool names.
2. For movement tools ('forward', 'backward'), ALWAYS use semantic 'speed_profile' ("slow", "medium", "fast") instead of numeric speed values, unless the user explicitly requested an exact number.
3. If the user did NOT specify how fast or with what speed to move, do NOT guess or invent a speed. Indicate that clarification is needed.
4. If the tool is 'stop', it requires no parameters.
5. Multi-action plans must contain at most {max_actions} sequential actions.
6. The robot never executes actions in parallel; actions execute strictly one after another.
7. Return ONLY a valid JSON object. Do not include markdown formatting or conversational filler outside the JSON.

JSON OUTPUT SCHEMA:
{{
  "plan_explanation": "Brief description of the planned action",
  "actions": [
    {{
      "action_id": "action_1",
      "tool": "forward | backward | stop | null",
      "parameters": {{
        "speed_profile": "slow | medium | fast",
        "steps": 1
      }}
    }}
  ]
}}

If a command is unsupported or invalid:
{{
  "plan_explanation": "Cannot perform requested action",
  "actions": [
    {{
      "action_id": "action_1",
      "tool": null,
      "reason": "Reason why the request cannot be fulfilled"
    }}
  ]
}}
"""


FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {
        "user": "Move forward slowly for 2 steps",
        "assistant": json.dumps({
            "plan_explanation": "Move forward slowly for 2 steps",
            "actions": [
                {
                    "action_id": "action_1",
                    "tool": "forward",
                    "parameters": {
                        "speed_profile": "slow",
                        "steps": 2
                    }
                }
            ]
        })
    },
    {
        "user": "Move forward",
        "assistant": json.dumps({
            "plan_explanation": "Move forward requested but speed is missing",
            "actions": [
                {
                    "action_id": "action_1",
                    "tool": "forward",
                    "parameters": {}
                }
            ]
        })
    },
    {
        "user": "Stop immediately",
        "assistant": json.dumps({
            "plan_explanation": "Halt all robot movement immediately",
            "actions": [
                {
                    "action_id": "action_1",
                    "tool": "stop",
                    "parameters": {}
                }
            ]
        })
    },
    {
        "user": "Move forward slowly then backward slowly",
        "assistant": json.dumps({
            "plan_explanation": "Move forward then backward at slow speed",
            "actions": [
                {
                    "action_id": "action_1",
                    "tool": "forward",
                    "parameters": {
                        "speed_profile": "slow",
                        "steps": 1
                    }
                },
                {
                    "action_id": "action_2",
                    "tool": "backward",
                    "parameters": {
                        "speed_profile": "slow",
                        "steps": 1
                    }
                }
            ]
        })
    },
    {
        "user": "Fly to the moon",
        "assistant": json.dumps({
            "plan_explanation": "Flying is not supported",
            "actions": [
                {
                    "action_id": "action_1",
                    "tool": None,
                    "reason": "Tool 'fly' is not supported."
                }
            ]
        })
    }
]


def build_tool_calling_prompt(
    user_message: str,
    previous_history: Optional[List[Dict[str, Any]]] = None,
) -> List[ChatMessage]:
    """
    Constructs the prompt for Qwen 2.5 3B tool calling.
    Injects available tool JSON schemas and few-shot examples.
    """
    schemas = get_available_tool_schemas()
    formatted_schemas = json.dumps(schemas, indent=2)

    system_content = TOOL_CALLING_SYSTEM_PROMPT.format(
        tool_schemas=formatted_schemas,
        max_actions=settings.MAX_ACTIONS_PER_PLAN,
    )

    messages: List[ChatMessage] = [
        ChatMessage(role="system", content=system_content)
    ]

    # Inject few-shot examples
    for eg in FEW_SHOT_EXAMPLES:
        messages.append(ChatMessage(role="user", content=eg["user"]))
        messages.append(ChatMessage(role="assistant", content=eg["assistant"]))

    # Inject recent conversation history if provided
    if previous_history:
        for msg in previous_history[-4:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append(ChatMessage(role=role, content=content))

    # Current user message
    messages.append(ChatMessage(role="user", content=user_message))

    return messages
