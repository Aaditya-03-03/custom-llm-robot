from enum import Enum
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class ControlAuthority(str, Enum):
    MANUAL = "MANUAL"
    AI = "AI"


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

    # Stage 10 Fine-Tuning & Local Model Deployment settings
    LLM_FINE_TUNED_MODEL_NAME: str = "ioft-qwen25-3b-v1"
    LLM_ADAPTER_PATH: Optional[str] = None
    USE_FINE_TUNED_MODEL: bool = False

    # MongoDB Conversation Memory settings
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "custom_llm_robot"
    MONGODB_CONVERSATIONS_COLLECTION: str = "conversations"
    MONGODB_MESSAGES_COLLECTION: str = "messages"
    CONVERSATION_HISTORY_LIMIT: int = 10

    # Stage 6 Physical Execution & ESP32 Locomotion settings
    ESP32_HOST: str = "rioft.local"
    ESP32_PORT: int = 8888
    ESP32_TIMEOUT_SECONDS: float = 0.5

    ROBOT_STEP_DURATION_SECONDS: float = 0.5
    ROBOT_DEFAULT_STEP_COUNT: int = 1
    ROBOT_COMMAND_REFRESH_INTERVAL_SECONDS: float = 0.25
    ROBOT_COMMAND_WATCHDOG_MS: int = 1000

    ROBOT_DEFAULT_AUTHORITY: ControlAuthority = ControlAuthority.MANUAL
    ROBOT_PIVOT_TURNS_VERIFIED: bool = False

    # Stage 7 LLM Tool Calling & Action Planning settings
    ROBOT_SPEED_SLOW: int = 20
    ROBOT_SPEED_MEDIUM: int = 40
    ROBOT_SPEED_FAST: int = 70
    MAX_ACTIONS_PER_PLAN: int = 3

    # Stage 8 Robot State Awareness & Telemetry settings
    ROBOT_STATE_STALE_AFTER_SECONDS: float = 5.0
    ROBOT_HEARTBEAT_INTERVAL_SECONDS: float = 2.0

    # Stage 9 Closed-Loop Execution, Verification & Recovery settings
    ROBOT_VERIFICATION_ENABLED: bool = True
    ROBOT_VERIFICATION_TIMEOUT_SECONDS: float = 2.0
    ROBOT_RECOVERY_ENABLED: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
