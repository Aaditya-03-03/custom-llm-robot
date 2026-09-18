"""
Chat endpoint — Stage 3 (MongoDB conversation memory + RAG integration)

Physical file:  app/api/chat.py
Exposed URLs:
    POST   /api/v1/chat
    GET    /api/v1/chat/{session_id}/history
    DELETE /api/v1/chat/{session_id}

Message ordering guarantee (per implementation plan):
    1. Resolve/create session
    2. Fetch previous history  ← BEFORE current user message is saved
    3. Save current user message
    4. RAG retrieval
    5. Build prompt
    6. LLM generation
    7. Save assistant response
    8. Return ChatResponse

Failure behaviour:
    - MongoDB unavailable          → HTTP 503
    - Empty/whitespace session_id  → HTTP 400
    - LLM generation fails         → HTTP 502; user message is RETAINED in MongoDB
    - RAG failure                  → HTTP 500; user message is RETAINED in MongoDB
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.llm.factory import get_llm_provider
from app.llm.ollama_provider import OllamaProviderError
from app.llm.prompts import build_chat_prompt
from app.memory.database import get_mongo_client, is_mongodb_available
from app.memory.manager import ConversationManager
from app.rag.retriever import KnowledgeRetriever
from app.rag.vector_store import VectorStoreManager
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    HistoryResponse,
    MessageRecord,
)
from app.intent.normalizer import normalize_deterministic_intent
from app.intent.schemas import IntentCategory

logger = logging.getLogger("custom_llm_robot.api.chat")
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_manager() -> ConversationManager:
    """
    Return a ConversationManager backed by the active MongoClient singleton.
    Raises HTTP 503 if MongoDB is not available.
    """
    if not is_mongodb_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Conversation memory service is unavailable. "
                "MongoDB is not reachable. Please try again later."
            ),
        )
    client = get_mongo_client()
    return ConversationManager(client)


def _validate_session_id(session_id: str | None) -> str | None:
    """
    Validate session_id input per contract:
    - None → pass through (auto-generate in manager)
    - Non-empty string → pass through
    - Empty / whitespace → raise HTTP 400
    """
    if session_id is not None and not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "session_id must be a non-empty string or omitted entirely. "
                "Empty or whitespace-only session_id values are not accepted."
            ),
        )
    return session_id


# ---------------------------------------------------------------------------
# POST /api/v1/chat
# ---------------------------------------------------------------------------

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message and receive a context-aware response",
)
async def chat_endpoint(request: ChatRequest):
    """
    Stage 3 chat endpoint with MongoDB conversation memory and RAG retrieval.

    Correct message ordering:
    1. Resolve/create session
    2. Fetch previous history (BEFORE saving current message)
    3. Save current user message
    4. RAG retrieval
    5. Build prompt: System + RAG Context + Previous History + Current Message
    6. LLM generation
    7. Save assistant response
    8. Return ChatResponse with session_id
    """
    # --- Validate session_id upfront ---
    raw_session_id = _validate_session_id(request.session_id)

    # --- Guard: MongoDB must be available ---
    manager = _get_manager()

    # --- Step 1: Resolve / create session ---
    session_id = manager.resolve_session(raw_session_id)
    logger.info(f"[{session_id}] Chat request received: '{request.message[:80]}'")

    # --- Step 2: Capture cutoff timestamp and fetch previous history ---
    # The cutoff is captured NOW, before saving the current message.
    # This guarantees the current message cannot appear in the history
    # retrieved for the prompt.
    history_cutoff = datetime.now(timezone.utc)
    previous_history = manager.fetch_previous_history(
        session_id=session_id,
        before=history_cutoff,
    )
    logger.info(
        f"[{session_id}] Fetched {len(previous_history)} previous message(s) "
        f"(limit={settings.CONVERSATION_HISTORY_LIMIT})"
    )

    # --- Step 3: Save current user message ---
    # Saved BEFORE LLM generation so the message is persisted even if the
    # LLM call fails. MongoDB will have a truthful record.
    manager.save_user_message(session_id=session_id, content=request.message)
    logger.info(f"[{session_id}] User message saved to MongoDB.")

    # --- Step 4: RAG retrieval ---
    # Query ChromaDB using the existing Stage 2 retriever and threshold.
    # The retriever returns (answer, sources) for /knowledge/query — here we
    # only need the context chunks, so we use the vector store directly.
    rag_context: str | None = None
    rag_sources = []
    try:
        vector_store = VectorStoreManager()
        matches = vector_store.query(request.message, top_k=3)
        from app.rag.config import rag_settings

        relevant = [
            m for m in matches
            if m.get("score", 0) >= rag_settings.RAG_SIMILARITY_THRESHOLD
        ]
        logger.info(
            f"[{session_id}] RAG: {len(matches)} chunk(s) retrieved, "
            f"{len(relevant)} passed threshold ({rag_settings.RAG_SIMILARITY_THRESHOLD})"
        )

        if relevant:
            blocks = []
            for idx, m in enumerate(relevant, start=1):
                doc = m.get("metadata", {}).get("document", "unknown")
                blocks.append(
                    f"--- Chunk {idx} (Source: {doc}) ---\n{m['text']}"
                )
                rag_sources.append(
                    {"document": doc, "score": m.get("score")}
                )
            rag_context = "\n\n".join(blocks)
    except Exception as e:
        # RAG failure: user message is already saved. Propagate error.
        logger.error(
            f"[{session_id}] RAG retrieval failed: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG retrieval failed. Your message has been saved.",
        )

    # --- Step 5: Build prompt ---
    messages = build_chat_prompt(
        user_message=request.message,
        previous_history=previous_history,
        rag_context=rag_context,
    )
    logger.info(
        f"[{session_id}] Prompt assembled: {len(messages)} message(s), "
        f"RAG context={'yes' if rag_context else 'no'}"
    )

    # --- Step 6: LLM generation ---
    try:
        provider = get_llm_provider()
        answer = await provider.generate_response(messages)
    except OllamaProviderError as e:
        # LLM failure: user message is already saved. Do NOT create a fake
        # assistant message. Return error — truthful MongoDB record preserved.
        logger.error(f"[{session_id}] LLM generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM service error: {str(e)}. Your message has been saved.",
        )
    except Exception as e:
        logger.error(
            f"[{session_id}] Unexpected LLM error: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during response generation. Your message has been saved.",
        )

    # --- Step 7: Save assistant response ---
    # Only reached on successful generation.
    manager.save_assistant_message(session_id=session_id, content=answer)
    logger.info(f"[{session_id}] Assistant response saved to MongoDB.")

    # --- Step 8: Return response (with lightweight deterministic intent annotation if matched) ---
    det_intent = normalize_deterministic_intent(request.message)
    annotated_intent = det_intent if (det_intent and det_intent.category == IntentCategory.ROBOT_COMMAND) else None

    return ChatResponse(
        session_id=session_id,
        response=answer,
        intent=annotated_intent,
    )



# ---------------------------------------------------------------------------
# GET /api/v1/chat/{session_id}/history
# ---------------------------------------------------------------------------

@router.get(
    "/chat/{session_id}/history",
    response_model=HistoryResponse,
    summary="Retrieve full conversation history for a session",
)
async def get_session_history(session_id: str):
    """
    Return all stored messages for the given session, ordered oldest-first.

    Returns HTTP 404 if no session with the given session_id exists.
    Returns HTTP 503 if MongoDB is unavailable.
    """
    manager = _get_manager()
    exists, messages = manager.get_full_history(session_id)

    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    records = [
        MessageRecord(
            role=m["role"],
            content=m["content"],
            timestamp=m["timestamp"],
        )
        for m in messages
    ]
    logger.info(
        f"History request: session={session_id}, {len(records)} message(s) returned."
    )
    return HistoryResponse(session_id=session_id, messages=records)


# ---------------------------------------------------------------------------
# DELETE /api/v1/chat/{session_id}
# ---------------------------------------------------------------------------

@router.delete(
    "/chat/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a session and all its messages",
)
async def delete_session(session_id: str):
    """
    Delete the session and all associated messages from MongoDB.

    Returns HTTP 204 on success.
    Returns HTTP 404 if the session does not exist.
    Returns HTTP 503 if MongoDB is unavailable.
    """
    manager = _get_manager()
    deleted = manager.delete_session(session_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    logger.info(f"Session deleted: {session_id}")
