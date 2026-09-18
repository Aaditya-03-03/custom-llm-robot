"""
Pydantic schemas for direct commands and tool execution requests.
"""

from pydantic import BaseModel, Field

class CommandExecutionRequest(BaseModel):
    tool_name: str = Field(..., description="Name of registered robot tool")
    parameters: dict = Field(default_factory=dict, description="Tool execution parameters")

class CommandExecutionResponse(BaseModel):
    success: bool
    message: str
    result: dict = Field(default_factory=dict)
