import logging
from typing import Dict, Any, List, Tuple
from app.rag.config import rag_settings
from app.rag.vector_store import VectorStoreManager
from app.llm.prompts import RAG_SYSTEM_PROMPT
from app.llm.factory import get_llm_provider
from app.schemas.chat import ChatMessage

logger = logging.getLogger("custom_llm_robot.rag.retriever")

FALLBACK_RESPONSE = "The requested information was not found in the project knowledge base."

class KnowledgeRetriever:
    """
    RAG retrieval engine: Performs vector similarity search, enforces relevance thresholds,
    constructs prompt contexts, and dispatches to local LLM provider.
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager = None,
        similarity_threshold: float = rag_settings.RAG_SIMILARITY_THRESHOLD,
        top_k: int = rag_settings.RAG_TOP_K
    ):
        self.vector_store_manager = vector_store_manager or VectorStoreManager()
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

    async def query(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Query the knowledge base with a user question.
        Returns tuple of (answer_text, list_of_source_metadata).
        """
        logger.info(f"RAG Query: '{question}'")
        
        matches = self.vector_store_manager.query(question, top_k=self.top_k)
        logger.info(f"Retrieved {len(matches)} chunk(s) from ChromaDB.")

        if not matches:
            logger.info("No matching chunks returned from vector store.")
            return FALLBACK_RESPONSE, []

        # Filter matches by similarity threshold
        relevant_matches = [m for m in matches if m["score"] >= self.similarity_threshold]
        
        for m in matches:
            logger.debug(f"Chunk [{m['chunk_id']}] Score: {m['score']} (Relevant: {m['score'] >= self.similarity_threshold})")

        if not relevant_matches:
            logger.info(f"All retrieved chunks failed similarity threshold ({self.similarity_threshold}). Max score: {matches[0]['score']}")
            return FALLBACK_RESPONSE, []

        # Format source metadata
        sources = []
        for m in relevant_matches:
            meta = m["metadata"]
            sources.append({
                "document": meta.get("document", "unknown"),
                "chunk_id": m["chunk_id"],
                "score": m["score"]
            })

        # Format context block
        context_blocks = []
        for idx, m in enumerate(relevant_matches, start=1):
            doc_name = m["metadata"].get("document", "doc")
            context_blocks.append(f"--- Document Chunk {idx} (Source: {doc_name}) ---\n{m['text']}")

        joined_context = "\n\n".join(context_blocks)

        user_prompt = f"""Context from project documentation:
{joined_context}

User Question: {question}"""

        messages = [
            ChatMessage(role="system", content=RAG_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt)
        ]

        logger.info(f"Dispatching RAG prompt to LLM provider with {len(relevant_matches)} relevant chunk(s)...")
        provider = get_llm_provider()
        answer = await provider.generate_response(messages)

        logger.info("RAG answer generated successfully.")
        return answer, sources
