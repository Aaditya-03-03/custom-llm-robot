"""
CLI entry point for ingesting and updating document embeddings in ChromaDB.

Usage:
    python app/rag/ingest.py
    OR
    python -m app.rag.ingest
"""

import os
import sys
import logging

# Ensure the project root directory is in sys.path when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.rag.vector_store import VectorStoreManager
from app.rag.config import rag_settings

logger = logging.getLogger("custom_llm_robot.rag.ingest")

def run_ingestion():
    logger.info("=== Starting Knowledge Base Ingestion ===")
    logger.info(f"Target Documents Directory: {rag_settings.DOCUMENTS_DIR}")
    logger.info(f"Target Vector DB Directory: {rag_settings.VECTOR_DB_DIR}")

    try:
        manager = VectorStoreManager()
        stats = manager.sync_documents()
        logger.info(
            f"Ingestion completed successfully: "
            f"Added {stats['added']} file(s), Updated {stats['updated']} file(s), "
            f"Deleted {stats['deleted']} file(s). Total chunks inserted: {stats['total_chunks_inserted']}."
        )
        print(f"Ingestion complete: {stats}")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ingestion()
