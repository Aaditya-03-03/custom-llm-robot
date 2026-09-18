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
        logger.info(f"MockLLMProvider processing message: '{last_user_msg}'")
        return f"I am the IOFT Humanoid Robot AI (Mock Mode). Received: '{last_user_msg}'"

    async def check_health(self) -> bool:
        return True
