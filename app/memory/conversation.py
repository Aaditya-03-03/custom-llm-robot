"""
Managing ongoing chat history and state context for sessions.
"""

class ConversationMemory:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: list[dict] = []

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        return self.history
