import logging
from app.llm.base import BaseLLMProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.mock_provider import MockLLMProvider
from app.core.config import settings

logger = logging.getLogger("custom_llm_robot.factory")

def get_llm_provider() -> BaseLLMProvider:
    """
    Instantiate and return the configured LLM provider based on environment setting.
    Explicit choices: 'ollama' or 'mock'.
    """
    provider_name = settings.LLM_PROVIDER.lower().strip()
    
    if provider_name == "ollama":
        model_name = settings.LLM_FINE_TUNED_MODEL_NAME if settings.USE_FINE_TUNED_MODEL else settings.OLLAMA_MODEL
        logger.info(f"Initializing OllamaProvider (Model: {model_name}, URL: {settings.OLLAMA_BASE_URL}, FineTuned: {settings.USE_FINE_TUNED_MODEL})")
        return OllamaProvider(model=model_name)
    elif provider_name == "mock":
        logger.info("Initializing MockLLMProvider")
        return MockLLMProvider()
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER '{settings.LLM_PROVIDER}'. Supported options: 'ollama', 'mock'.")
