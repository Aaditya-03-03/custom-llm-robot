import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class RAGSettings(BaseSettings):
    DOCUMENTS_DIR: str = os.path.join("data", "documents")
    VECTOR_DB_DIR: str = os.path.join("data", "vector_db")
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    RAG_TOP_K: int = 3
    RAG_CHUNK_SIZE: int = 500
    RAG_CHUNK_OVERLAP: int = 50
    RAG_SIMILARITY_THRESHOLD: float = 0.35
    RAG_AUTO_INGEST: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

rag_settings = RAGSettings()
