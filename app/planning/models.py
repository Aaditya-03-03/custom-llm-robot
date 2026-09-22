"""
Pydantic data models for Stage 7 Action Planning, Tool Calling, and Assistant API.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from app.schemas.commands import PhysicalExecutionResponse


class AssistantStatus(str, Enum):
    ASSISTANT_EXECUTED = "ASSISTANT_EXECUTED"
    ASSISTANT_CLARIFICATION_REQUIRED = "ASSISTANT_CLARIFICATION_REQUIRED"
    ASSISTANT_OWNERSHIP_REJECTED = "ASSISTANT_OWNERSHIP_REJECTED"
    ASSISTANT_ROBOT_UNAVAILABLE = "ASSISTANT_ROBOT_UNAVAILABLE"
    ASSISTANT_UNSUPPORTED = "ASSISTANT_UNSUPPORTED"
    ASSISTANT_VALIDATION_FAILED = "ASSISTANT_VALIDATION_FAILED"
    ASSISTANT_EXECUTION_FAILED = "ASSISTANT_EXECUTION_FAILED"
    ASSISTANT_PARTIAL_FAILURE = "ASSISTANT_PARTIAL_FAILURE"


class ToolCall(BaseModel):
    action_id: str = Field(..., description="Unique action identifier within plan (e.g. action_1)")
    tool: Optional[str] = Field(default=None, description="Canonical tool name (or None if unsupported)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Resolved physical parameters")
    speed_profile: Optional[str] = Field(default=None, description="Semantic speed profile (slow/medium/fast)")
    requested_speed: Optional[int] = Field(default=None, description="Explicit requested numeric speed")
    steps: Optional[int] = Field(default=None, description="Steps count (defaults to 1 if omitted for movement)")
    needs_clarification: bool = Field(default=False, description="True if required parameters are missing")
    missing: List[str] = Field(default_factory=list, description="Names of missing required parameters")
    reason: Optional[str] = Field(default=None, description="Reason if tool call cannot be fulfilled")


class ExecutionPlan(BaseModel):
    plan_id: str = Field(..., description="Unique UUID for this execution plan")
    actions: List[ToolCall] = Field(default_factory=list, description="Sequential actions in plan (max 3)")
    plan_explanation: Optional[str] = Field(default=None, description="High-level natural language description")


class AssistantRequest(BaseModel):
    message: str = Field(..., description="User natural language request or command")
    session_id: Optional[str] = Field(default=None, description="Optional MongoDB conversation session UUID")


class AssistantResponse(BaseModel):
    request_id: str = Field(..., description="Unique request tracking ID")
    session_id: str = Field(..., description="Active session ID")
    plan_id: Optional[str] = Field(default=None, description="Plan tracking ID if a plan was generated")
    status: AssistantStatus = Field(..., description="High-level assistant processing status")
    response_text: str = Field(..., description="Conversational or explanatory response text")
    executed: bool = Field(default=False, description="True if all planned physical actions were executed")
    plan: Optional[ExecutionPlan] = Field(default=None, description="Parsed and validated execution plan")
    execution_results: List[PhysicalExecutionResponse] = Field(
        default_factory=list, description="Stage 6 execution responses for each executed action"
    )
    completed_actions: List[str] = Field(
        default_factory=list, description="List of successfully executed action_ids"
    )
    failed_action: Optional[str] = Field(
        default=None, description="Action ID of the action that failed, if any"
    )
    remaining_actions_aborted: List[str] = Field(
        default_factory=list, description="List of action_ids aborted due to an earlier failure"
    )
    errors: List[str] = Field(default_factory=list, description="Any validation or execution errors")
    execution_records: List[Dict[str, Any]] = Field(
        default_factory=list, description="Stage 9 closed-loop execution and verification records"
    )
