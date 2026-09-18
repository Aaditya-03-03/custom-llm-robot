from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "IOFT Humanoid Robot AI Server"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # LLM settings
    LLM_PROVIDER: str = "ollama"  # "ollama" or "mock"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"
    OLLAMA_TIMEOUT_SECONDS: float = 60.0

    # MongoDB Conversation Memory settings
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "custom_llm_robot"
    MONGODB_CONVERSATIONS_COLLECTION: str = "conversations"
    MONGODB_MESSAGES_COLLECTION: str = "messages"
    CONVERSATION_HISTORY_LIMIT: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
