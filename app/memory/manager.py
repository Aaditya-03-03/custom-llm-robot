"""
High-level Conversation Manager.

Orchestrates the complete session lifecycle used by the chat endpoint:

    1. Resolve or create session
    2. Fetch previous history (BEFORE saving current user message)
    3. Save current user message
    4. (Caller: RAG retrieval + LLM generation)
    5. Save assistant response
    6. Return previous history + session_id

This ordering ensures:
- The current user message NEVER appears in the fetched history.
- The user message is persisted BEFORE LLM generation, so it survives an LLM failure.
- The assistant message is only written if LLM generation succeeds.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

from pymongo import MongoClient

from app.core.config import settings
from app.memory.repository import ConversationRepository, MessageRepository

logger = logging.getLogger("custom_llm_robot.memory.manager")


class ConversationManager:
    """
    High-level API for session memory operations.

    All methods are synchronous (underlying repository uses PyMongo sync driver).
    """

    def __init__(self, client: MongoClient) -> None:
        self._conversations = ConversationRepository(client)
        self._messages = MessageRepository(client)

    # ------------------------------------------------------------------ #
    #  Session Resolution                                                   #
    # ------------------------------------------------------------------ #

    def resolve_session(self, session_id: str | None) -> str:
        """
        Resolve an incoming session_id according to the contract:

        - None / omitted  → auto-generate a UUID session.
        - Non-empty string, session exists  → reuse.
        - Non-empty string, session does NOT exist → create with that ID.

        Returns the resolved session_id string.

        NOTE: Empty/whitespace validation is handled at the API layer (HTTP 400).
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
            logger.info(f"Auto-generated session_id: {session_id}")

        self._conversations.get_or_create(session_id)
        return session_id

    # ------------------------------------------------------------------ #
    #  Core Chat Flow                                                       #
    # ------------------------------------------------------------------ #

    def fetch_previous_history(
        self,
        session_id: str,
        before: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Fetch up to CONVERSATION_HISTORY_LIMIT messages stored BEFORE `before`.

        `before` must be set to the timestamp captured immediately before the
        current user message is saved, ensuring the current message never
        appears in the returned history.
        """
        limit = settings.CONVERSATION_HISTORY_LIMIT
        history = self._messages.get_previous_history(
            session_id=session_id,
            limit=limit,
            before=before,
        )
        logger.info(
            f"Fetched {len(history)} previous message(s) for session {session_id} "
            f"(limit={limit})"
        )
        return history

    def save_user_message(self, session_id: str, content: str) -> None:
        """
        Persist the current user message.

        Called AFTER fetch_previous_history so the current message cannot
        appear in the history passed to the LLM prompt.

        Persisted BEFORE LLM generation so the message survives an LLM failure.
        """
        self._messages.save(session_id=session_id, role="user", content=content)
        self._conversations.touch(session_id)
        logger.debug(f"Saved user message for session {session_id}")

    def save_assistant_message(self, session_id: str, content: str) -> None:
        """
        Persist the assistant response.

        Called ONLY after successful LLM generation.
        If LLM generation fails, this is never called, leaving MongoDB with
        only the user message — which is the truthful record.
        """
        self._messages.save(session_id=session_id, role="assistant", content=content)
        self._conversations.touch(session_id)
        logger.debug(f"Saved assistant message for session {session_id}")

    # ------------------------------------------------------------------ #
    #  History & Session Management Endpoints                              #
    # ------------------------------------------------------------------ #

    def get_full_history(self, session_id: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Return (session_exists, all_messages) for the history endpoint.
        Messages are sorted chronologically.
        """
        exists = self._conversations.exists(session_id)
        if not exists:
            return False, []
        messages = self._messages.get_all(session_id)
        return True, messages

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its messages.
        Returns True if the session existed and was deleted, False otherwise.
        """
        if not self._conversations.exists(session_id):
            return False
        self._messages.delete_all(session_id)
        self._conversations.delete(session_id)
        logger.info(f"Deleted session and all messages: {session_id}")
        return True
