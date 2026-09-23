import logging
import httpx
from typing import List, Any
from app.llm.base import BaseLLMProvider
from app.schemas.chat import ChatMessage
from app.core.config import settings

logger = logging.getLogger("custom_llm_robot.ollama")

class OllamaProviderError(Exception):
    """Custom exception raised when Ollama service fails or returns an error."""
    pass

class OllamaProvider(BaseLLMProvider):
    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model: str = settings.OLLAMA_MODEL,
        timeout: float = settings.OLLAMA_TIMEOUT_SECONDS
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def generate_response(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        url = f"{self.base_url}/api/chat"
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": False
        }
        if "options" in kwargs and kwargs["options"]:
            payload["options"] = kwargs["options"]

        logger.info(f"Dispatching prompt to Ollama model '{self.model}' at {url}")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                content = data.get("message", {}).get("content", "")
                if not content:
                    logger.warning(f"Ollama returned empty response payload: {data}")
                return content
            except httpx.ConnectError as e:
                logger.error(f"Cannot connect to Ollama service at {self.base_url}: {e}")
                raise OllamaProviderError(f"Ollama local service is unreachable at {self.base_url}. Ensure Ollama is running.") from e
            except httpx.TimeoutException as e:
                logger.error(f"Ollama request timed out after {self.timeout}s: {e}")
                raise OllamaProviderError(f"Ollama request timed out after {self.timeout}s (model may be cold starting). Please retry.") from e
            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama HTTP status error {e.response.status_code}: {e.response.text}")
                raise OllamaProviderError(f"Ollama API error ({e.response.status_code}): {e.response.text}") from e
            except Exception as e:
                err_msg = str(e) or type(e).__name__
                logger.error(f"Unexpected error during Ollama inference: {err_msg}")
                raise OllamaProviderError(f"Failed to generate response from Ollama: {err_msg}") from e

    async def check_health(self) -> bool:
        """Check if Ollama service is reachable and responsive."""
        url = f"{self.base_url}/api/tags"
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(url)
                return response.status_code == 200
            except Exception as e:
                logger.warning(f"Ollama health check failed for {url}: {e}")
                return False
