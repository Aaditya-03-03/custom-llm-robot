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

        # If invoked for intent extraction, return mock JSON matching intent schema
        if "intent detection engine" in sys_msg.lower():
            lower_user = last_user_msg.lower()
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
