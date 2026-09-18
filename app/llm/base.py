from abc import ABC, abstractmethod
from typing import List, Dict, Any
from app.schemas.chat import ChatMessage

class BaseLLMProvider(ABC):
    """
    Abstract interface for LLM inference providers.
    Ensures total modularity so Ollama, llama.cpp, vLLM, or other engines
    can be swapped without altering API routers or control layers.
    """

    @abstractmethod
    async def generate_response(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        """Generate response from LLM given a sequence of chat messages."""
        pass

    @abstractmethod
    async def check_health(self) -> bool:
        """Check if the backend LLM service is online and healthy."""
        pass
