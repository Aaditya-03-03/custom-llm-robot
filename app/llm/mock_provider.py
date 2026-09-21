import json
import logging
from typing import List, Any
from app.llm.base import BaseLLMProvider
from app.schemas.chat import ChatMessage

logger = logging.getLogger("custom_llm_robot.mock")


class MockLLMProvider(BaseLLMProvider):
    """
    Mock LLM Provider for explicit testing and local UI development
    when set via LLM_PROVIDER=mock.
    """

    async def generate_response(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        last_user_msg = next((msg.content for msg in reversed(messages) if msg.role == "user"), "Hello")
        sys_msg = next((msg.content for msg in messages if msg.role == "system"), "")
        logger.info(f"MockLLMProvider processing message: '{last_user_msg}'")

        lower_user = last_user_msg.lower()

        # 1. Stage 7 Action Planning / Tool Calling
        if "robot action planning engine" in sys_msg.lower():
            if "corrupt" in lower_user:
                return "Not a valid JSON response from model {broken..."
            elif "contradictory" in lower_user:
                return json.dumps({
                    "plan_explanation": "Contradictory speed",
                    "actions": [{
                        "action_id": "action_1",
                        "tool": "forward",
                        "parameters": {"speed": 30, "speed_profile": "slow", "steps": 1}
                    }]
                })
            elif "invalid profile" in lower_user or "turbo" in lower_user:
                return json.dumps({
                    "plan_explanation": "Invalid speed profile",
                    "actions": [{
                        "action_id": "action_1",
                        "tool": "forward",
                        "parameters": {"speed_profile": "super_fast", "steps": 1}
                    }]
                })
            elif "too many" in lower_user:
                return json.dumps({
                    "plan_explanation": "Exceeds max actions",
                    "actions": [
                        {"action_id": "action_1", "tool": "forward", "parameters": {"speed_profile": "slow", "steps": 1}},
                        {"action_id": "action_2", "tool": "backward", "parameters": {"speed_profile": "slow", "steps": 1}},
                        {"action_id": "action_3", "tool": "forward", "parameters": {"speed_profile": "slow", "steps": 1}},
                        {"action_id": "action_4", "tool": "stop", "parameters": {}},
                    ]
                })
            elif "move_joint" in lower_user or "bypass" in lower_user:
                return json.dumps({
                    "plan_explanation": "Direct joint bypass proposal",
                    "actions": [{
                        "action_id": "action_1",
                        "tool": "move_joint",
                        "parameters": {"joint_id": 1, "angle": 45}
                    }]
                })
            elif "fly" in lower_user or "jump" in lower_user or "dance" in lower_user:
                return json.dumps({
                    "plan_explanation": "Unsupported request",
                    "actions": [{
                        "action_id": "action_1",
                        "tool": None,
                        "reason": f"Tool '{last_user_msg}' is not supported."
                    }]
                })
            elif "999" in lower_user or "excessive" in lower_user:
                return json.dumps({
                    "plan_explanation": "Excessive speed request",
                    "actions": [{
                        "action_id": "action_1",
                        "tool": "forward",
                        "parameters": {"speed": 999, "steps": 1}
                    }]
                })
            elif "stop" in lower_user:
                return json.dumps({
                    "plan_explanation": "Stop movement",
                    "actions": [{
                        "action_id": "action_1",
                        "tool": "stop",
                        "parameters": {}
                    }]
                })
            elif "then" in lower_user or "multi" in lower_user or ("forward" in lower_user and "backward" in lower_user):
                return json.dumps({
                    "plan_explanation": "Multi-action sequence",
                    "actions": [
                        {
                            "action_id": "action_1",
                            "tool": "forward",
                            "parameters": {"speed_profile": "slow", "steps": 1}
                        },
                        {
                            "action_id": "action_2",
                            "tool": "backward",
                            "parameters": {"speed_profile": "slow", "steps": 1}
                        }
                    ]
                })
            elif "do that again" in lower_user or "repeat" in lower_user:
                return json.dumps({
                    "plan_explanation": "Repeat action from context",
                    "actions": [{
                        "action_id": "action_1",
                        "tool": "repeat",
                        "parameters": {}
                    }]
                })
            elif "slow" in lower_user:
                steps = 2 if "2" in lower_user else 1
                tool = "backward" if "backward" in lower_user else "forward"
                return json.dumps({
                    "plan_explanation": f"Move {tool} slowly",
                    "actions": [{
                        "action_id": "action_1",
                        "tool": tool,
                        "parameters": {"speed_profile": "slow", "steps": steps}
                    }]
                })
            elif "fast" in lower_user:
                tool = "backward" if "backward" in lower_user else "forward"
                return json.dumps({
                    "plan_explanation": f"Move {tool} fast",
                    "actions": [{
                        "action_id": "action_1",
                        "tool": tool,
                        "parameters": {"speed_profile": "fast", "steps": 1}
                    }]
                })
            elif "medium" in lower_user:
                tool = "backward" if "backward" in lower_user else "forward"
                return json.dumps({
                    "plan_explanation": f"Move {tool} at medium speed",
                    "actions": [{
                        "action_id": "action_1",
                        "tool": tool,
                        "parameters": {"speed_profile": "medium", "steps": 1}
                    }]
                })
            elif "forward" in lower_user or "backward" in lower_user:
                # Speed missing!
                tool = "backward" if "backward" in lower_user else "forward"
                return json.dumps({
                    "plan_explanation": f"Move {tool} requested without speed",
                    "actions": [{
                        "action_id": "action_1",
                        "tool": tool,
                        "parameters": {}
                    }]
                })
            else:
                return json.dumps({
                    "plan_explanation": "General conversational response",
                    "actions": []
                })

        # 2. Stage 4 Intent Extraction Mock
        if "intent detection engine" in sys_msg.lower():
            if "corrupt" in lower_user:
                return "Not a valid JSON response from model"
            elif "fly" in lower_user or "jump" in lower_user:
                return '{"category": "unsupported", "action": null, "parameters": null, "error_message": "Action unsupported"}'
            elif "forward" in lower_user:
                return '{"category": "robot_command", "action": "forward", "parameters": {"speed": 50, "steps": 3}, "error_message": null}'
            elif "backward" in lower_user:
                return '{"category": "robot_command", "action": "backward", "parameters": null, "error_message": null}'
            elif "left" in lower_user:
                return '{"category": "robot_command", "action": "left", "parameters": null, "error_message": null}'
            elif "right" in lower_user:
                return '{"category": "robot_command", "action": "right", "parameters": null, "error_message": null}'
            elif "stand" in lower_user:
                return '{"category": "robot_command", "action": "stand", "parameters": null, "error_message": null}'
            elif "sit" in lower_user:
                return '{"category": "robot_command", "action": "sit", "parameters": null, "error_message": null}'
            elif "stop" in lower_user:
                return '{"category": "robot_command", "action": "stop", "parameters": {"emergency": false}, "error_message": null}'
            elif "status" in lower_user:
                return '{"category": "robot_command", "action": "status", "parameters": null, "error_message": null}'
            elif "calibrate" in lower_user:
                return '{"category": "robot_command", "action": "calibrate", "parameters": null, "error_message": null}'
            elif "manual" in lower_user:
                return '{"category": "robot_command", "action": "manual", "parameters": null, "error_message": null}'
            else:
                return '{"category": "conversation", "action": null, "parameters": null, "error_message": null}'

        return f"I am the IOFT Humanoid Robot AI (Mock Mode). Received: '{last_user_msg}'"

    async def check_health(self) -> bool:
        return True
