"""
Tool registry for binding tool functions with LLM schema definitions.
"""

class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name: str, func):
        self._tools[name] = func

    def get_tool(self, name: str):
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())
