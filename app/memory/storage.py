"""
Persistent storage driver for storing conversation logs and agent state.
"""

class MemoryStorage:
    def save_session(self, session_id: str, data: dict):
        pass

    def load_session(self, session_id: str) -> dict:
        return {}
