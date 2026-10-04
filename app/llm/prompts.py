"""
Centralized system prompt definitions for IOFT Humanoid Robot.
"""

# ---------------------------------------------------------------------------
# Chat system prompt — used by POST /api/v1/chat
# ---------------------------------------------------------------------------
CHAT_SYSTEM_PROMPT = """\
You are a versatile, intelligent AI assistant for the IOFT Humanoid Robot project.

Your capabilities:
- You answer general knowledge, science, programming, everyday, and conversational questions directly and helpfully.
- You can explain, configure, and help operators control the IOFT Humanoid Robot.
- When asked open-ended questions like "how can you help me" or "what can you do", explain that you can help with both general questions/coding AND assist with the IOFT Humanoid Robot's controls and features.
- Never claim to have performed a physical robot action unless a robot-control system has confirmed it.

IMPORTANT — Project-Specific Information:
If a user asks about specific IOFT robot hardware, software, wiring, components, servo models, commands, or implementation details:
  - Base your answers on the PROJECT KNOWLEDGE CONTEXT provided in this conversation.
  - If a specific project hardware/software detail is not in the context, clearly state:
    "That specific detail is not available in the project knowledge base."
  - Do NOT invent or guess project-specific hardware facts.

General Knowledge:
You are fully capable of answering general knowledge, technical, programming, and conversational questions directly and naturally using your broad knowledge base.
"""

# ---------------------------------------------------------------------------
# RAG-only system prompt — used by POST /api/v1/knowledge/query
# (standalone knowledge base queries, no conversation context)
# ---------------------------------------------------------------------------
RAG_SYSTEM_PROMPT = """\
You are the AI assistant for IOFT's humanoid robot.

Answer the user's question strictly using the provided context below.
- Be concise, accurate, and direct.
- Do NOT invent or hallucinate project-specific facts that are not present in the context.
- If the context does not contain enough information to answer the question, state clearly: \
"The requested information was not found in the project knowledge base."
"""


# ---------------------------------------------------------------------------
# Prompt builder — assembles the full message list for chat endpoint
# ---------------------------------------------------------------------------
from typing import List, Optional, Dict, Any
from app.schemas.chat import ChatMessage


def build_chat_prompt(
    user_message: str,
    previous_history: List[Dict[str, Any]],
    rag_context: Optional[str] = None,
) -> List[ChatMessage]:
    """
    Assemble the ordered message list sent to the LLM:

        [system]
        [user]    PROJECT KNOWLEDGE CONTEXT (if RAG returned relevant chunks)
        [user/assistant] ... previous conversation history ...
        [user]    current user message

    Args:
        user_message:     The current user's input text.
        previous_history: Messages fetched from MongoDB BEFORE the current
                          user message was saved (role + content dicts).
        rag_context:      Formatted string of relevant document chunks,
                          or None if no relevant chunks passed the threshold.

    Returns:
        List of ChatMessage objects ready for BaseLLMProvider.generate_response().
    """
    messages: List[ChatMessage] = []

    # 1. System prompt
    messages.append(ChatMessage(role="system", content=CHAT_SYSTEM_PROMPT))

    # 2. RAG context block (injected as a user turn so models without a
    #    system-context slot still receive it correctly)
    if rag_context:
        context_block = (
            "PROJECT KNOWLEDGE CONTEXT\n"
            "(Use only this information for project-specific questions)\n\n"
            f"{rag_context}"
        )
        messages.append(ChatMessage(role="user", content=context_block))
        messages.append(
            ChatMessage(
                role="assistant",
                content="Understood. I will use the project knowledge context above to answer project-specific questions.",
            )
        )

    # 3. Previous conversation history
    for msg in previous_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append(ChatMessage(role=role, content=content))

    # 4. Current user message
    messages.append(ChatMessage(role="user", content=user_message))

    return messages
