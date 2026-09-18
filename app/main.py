"""
Main entry point for the IOFT Humanoid Robot AI Server.
"""

import os
import sys

# Ensure the project root directory is in sys.path when running app/main.py directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.core.logging import RequestIDMiddleware, logger
from app.api import health, chat, knowledge
from app.rag.config import rag_settings
from app.memory.database import connect_mongodb, disconnect_mongodb



@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        f"Starting {settings.APP_NAME} in "
        f"{'DEBUG' if settings.DEBUG else 'PRODUCTION'} mode."
    )
    logger.info(f"Active LLM Provider: {settings.LLM_PROVIDER}")

    # --- MongoDB initialisation ---
    # The application may start even if MongoDB is offline.
    # If unavailable, /api/v1/chat will return HTTP 503 until it recovers.
    connect_mongodb()

    # --- RAG auto-ingest (optional) ---
    if rag_settings.RAG_AUTO_INGEST:
        logger.info("RAG_AUTO_INGEST is enabled. Syncing documents with ChromaDB...")
        try:
            from app.rag.vector_store import VectorStoreManager
            manager = VectorStoreManager()
            stats = manager.sync_documents()
            logger.info(f"Auto-ingestion completed: {stats}")
        except Exception as e:
            logger.error(f"Auto-ingestion failed during startup: {e}")
    else:
        logger.info(
            "RAG_AUTO_INGEST is false. Using existing persistent vector database."
        )

    yield

    # --- Shutdown ---
    disconnect_mongodb()
    logger.info(f"Shutting down {settings.APP_NAME}.")


app = FastAPI(
    title=settings.APP_NAME,
    description="Modular local LLM-powered AI intelligence server for IOFT Humanoid Robot.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Attach Request ID middleware
app.add_middleware(RequestIDMiddleware)

# Include API Routers under /api/v1
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(knowledge.router, tags=["Knowledge"])


@app.get("/", include_in_schema=False)
async def root():
    return {
        "status": "online",
        "message": f"Welcome to {settings.APP_NAME}",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
