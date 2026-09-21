"""
Assistant API Router for Stage 7 LLM Tool Calling & Controlled Robot Action Planning.
Exposes POST /api/v1/assistant.
Connects conversation memory, Qwen 2.5 3B tool calling, deterministic parameter resolution,
all-or-nothing plan pre-validation, Stage 6 authority, and strictly sequential physical execution.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, status

from app.core.config import settings, ControlAuthority
from app.llm.factory import get_llm_provider
from app.llm.tool_calling import build_tool_calling_prompt
from app.memory.database import get_mongo_client, is_mongodb_available
from app.memory.manager import ConversationManager
from app.planning.models import (
    AssistantRequest,
    AssistantResponse,
    AssistantStatus,
    ExecutionPlan,
    ToolCall,
)
from app.planning.parser import parse_execution_plan
from app.planning.validator import pre_validate_execution_plan
from app.execution.adapter import default_execution_adapter
from app.execution.authority import default_authority_manager
from app.schemas.commands import PhysicalExecutionRequest, PhysicalExecutionResponse

logger = logging.getLogger("custom_llm_robot.api.assistant")
router = APIRouter()


def _get_conversation_manager() -> Optional[ConversationManager]:
    """Return ConversationManager if MongoDB is available, else None."""
    if is_mongodb_available():
        client = get_mongo_client()
        if client:
            return ConversationManager(client)
    return None


def _log_assistant_audit(
    request_id: str,
    session_id: str,
    user_message: str,
    plan: Optional[ExecutionPlan],
    status_val: AssistantStatus,
    completed: List[str],
    failed: Optional[str],
    aborted: List[str],
    errors: List[str],
) -> None:
    """Audit logging to MongoDB assistant_audits collection."""
    client = get_mongo_client()
    if client and is_mongodb_available():
        try:
            db = client[settings.MONGODB_DB_NAME]
            record = {
                "request_id": request_id,
                "session_id": session_id,
                "user_message": user_message,
                "plan": plan.model_dump() if plan else None,
                "status": status_val.value,
                "completed_actions": completed,
                "failed_action": failed,
                "remaining_actions_aborted": aborted,
                "errors": errors,
                "timestamp": datetime.now(timezone.utc),
            }
            db["assistant_audits"].insert_one(record)
        except Exception as e:
            logger.warning(f"Failed to record assistant audit log: {e}")


@router.post(
    "/assistant",
    response_model=AssistantResponse,
    summary="Natural Language Assistant & Controlled Robot Action Planning",
    description=(
        "Translates user natural language instructions into structured, validated robot action plans. "
        "Enforces deterministic parameter resolution, all-or-nothing pre-validation, "
        "fail-closed authority gates, and strictly sequential physical execution."
    ),
)
async def assistant_endpoint(request: AssistantRequest):
    request_id = str(uuid.uuid4())
    user_msg = (request.message or "").strip()
    if not user_msg:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User message cannot be empty or whitespace.",
        )

    # 1. Resolve Session & Fetch Context
    manager = _get_conversation_manager()
    session_id = request.session_id
    previous_history: List[Dict[str, Any]] = []
    recent_context: Optional[Dict[str, Any]] = None

    if manager:
        session_id = manager.resolve_session(session_id)
        before_time = datetime.now(timezone.utc)
        previous_history = manager.fetch_previous_history(session_id, before=before_time)
        manager.save_user_message(session_id, user_msg)

        # Check last assistant audit for contextual references (e.g. 'do that again')
        client = get_mongo_client()
        if client:
            try:
                db = client[settings.MONGODB_DB_NAME]
                last_audit = db["assistant_audits"].find_one(
                    {"session_id": session_id, "status": AssistantStatus.ASSISTANT_EXECUTED.value},
                    sort=[("timestamp", -1)],
                )
                if last_audit and last_audit.get("plan") and last_audit["plan"].get("actions"):
                    last_act = last_audit["plan"]["actions"][-1]
                    recent_context = {
                        "tool": last_act.get("tool"),
                        "parameters": last_act.get("parameters", {}),
                    }
            except Exception as err:
                logger.warning(f"Could not read prior assistant audit context: {err}")
    else:
        if not session_id:
            session_id = str(uuid.uuid4())

    logger.info(f"Assistant request [{request_id}] (session={session_id}): '{user_msg}'")

    # 2. LLM Tool Calling Inference
    llm_provider = get_llm_provider()
    prompt_messages = build_tool_calling_prompt(
        user_message=user_msg,
        previous_history=previous_history,
    )

    try:
        raw_llm_output = await llm_provider.generate_response(prompt_messages)
    except Exception as e:
        logger.error(f"Assistant LLM generation failed: {e}")
        err_msg = f"LLM provider error: {str(e)}"
        _log_assistant_audit(
            request_id, session_id, user_msg, None,
            AssistantStatus.ASSISTANT_UNSUPPORTED, [], None, [], [err_msg]
        )
        return AssistantResponse(
            request_id=request_id,
            session_id=session_id,
            status=AssistantStatus.ASSISTANT_UNSUPPORTED,
            response_text="I encountered an error while analyzing your request.",
            executed=False,
            errors=[err_msg],
        )

    # 3. Strict Parsing & Parameter Resolution
    plan, parse_err = parse_execution_plan(
        raw_response=raw_llm_output,
        recent_context=recent_context,
        user_message=user_msg,
    )

    if parse_err:
        logger.warning(f"Plan parsing failed for [{request_id}]: {parse_err}")
        _log_assistant_audit(
            request_id, session_id, user_msg, plan,
            AssistantStatus.ASSISTANT_VALIDATION_FAILED, [], None, [], [parse_err]
        )
        return AssistantResponse(
            request_id=request_id,
            session_id=session_id,
            plan_id=plan.plan_id,
            status=AssistantStatus.ASSISTANT_VALIDATION_FAILED,
            response_text=f"Failed to generate a valid plan: {parse_err}",
            executed=False,
            plan=plan,
            errors=[parse_err],
        )

    # If conversational with 0 actions
    if not plan.actions:
        resp_text = plan.plan_explanation or "I understood your message, but no robot action was requested."
        if manager:
            manager.save_assistant_message(session_id, resp_text)
        return AssistantResponse(
            request_id=request_id,
            session_id=session_id,
            plan_id=plan.plan_id,
            status=AssistantStatus.ASSISTANT_EXECUTED,
            response_text=resp_text,
            executed=False,
            plan=plan,
        )

    # 4. Plan Pre-Validation
    val_result = pre_validate_execution_plan(plan)

    # 4a. Distinguish Missing Required Parameter -> Early Clarification (No Stage 5 / Stage 6 calls)
    if val_result.needs_clarification:
        missing_str = ", ".join(val_result.missing_parameters)
        tool_name = plan.actions[0].tool if plan.actions else "movement"
        clarification_msg = (
            f"Please clarify the speed for '{tool_name}' (e.g. slow, medium, or fast)."
        )
        if manager:
            manager.save_assistant_message(session_id, clarification_msg)
        _log_assistant_audit(
            request_id, session_id, user_msg, plan,
            AssistantStatus.ASSISTANT_CLARIFICATION_REQUIRED, [], None, [], val_result.errors
        )
        return AssistantResponse(
            request_id=request_id,
            session_id=session_id,
            plan_id=plan.plan_id,
            status=AssistantStatus.ASSISTANT_CLARIFICATION_REQUIRED,
            response_text=clarification_msg,
            executed=False,
            plan=plan,
            errors=val_result.errors,
        )

    # 4b. Distinguish Invalid Parameter / Unsupported Tool -> Validation Failed (Zero Packets)
    if not val_result.is_valid:
        error_summary = "; ".join(val_result.errors)
        resp_text = f"Action plan validation rejected: {error_summary}"
        if manager:
            manager.save_assistant_message(session_id, resp_text)
        _log_assistant_audit(
            request_id, session_id, user_msg, plan,
            AssistantStatus.ASSISTANT_VALIDATION_FAILED, [], None, [], val_result.errors
        )
        return AssistantResponse(
            request_id=request_id,
            session_id=session_id,
            plan_id=plan.plan_id,
            status=AssistantStatus.ASSISTANT_VALIDATION_FAILED,
            response_text=resp_text,
            executed=False,
            plan=plan,
            errors=val_result.errors,
        )

    # 5. Execution Authority Gate
    current_auth = default_authority_manager.current_authority
    if current_auth == ControlAuthority.MANUAL:
        auth_err = "AI server currently in MANUAL authority mode. Physical execution rejected."
        logger.warning(f"Plan rejected by authority gate [{request_id}]: MANUAL mode active")
        if manager:
            manager.save_assistant_message(session_id, auth_err)
        _log_assistant_audit(
            request_id, session_id, user_msg, plan,
            AssistantStatus.ASSISTANT_OWNERSHIP_REJECTED, [], None, [], [auth_err]
        )
        return AssistantResponse(
            request_id=request_id,
            session_id=session_id,
            plan_id=plan.plan_id,
            status=AssistantStatus.ASSISTANT_OWNERSHIP_REJECTED,
            response_text=auth_err,
            executed=False,
            plan=plan,
            errors=[auth_err],
        )

    # 6. Strictly Sequential Physical Execution
    # Each action must completely return its Stage 6 execution result before the next action begins
    execution_results: List[PhysicalExecutionResponse] = []
    completed_actions: List[str] = []
    failed_action: Optional[str] = None
    remaining_actions_aborted: List[str] = []
    execution_errors: List[str] = []
    all_successful = True

    for idx, action in enumerate(plan.actions):
        logger.info(
            f"Executing plan action {idx+1}/{len(plan.actions)}: "
            f"action_id='{action.action_id}', tool='{action.tool}', params={action.parameters}"
        )
        exec_req = PhysicalExecutionRequest(
            tool_name=action.tool,
            parameters=action.parameters,
        )
        
        # Strictly sequential execution: awaits ACK/result completely
        exec_resp = await default_execution_adapter.execute_command(exec_req)
        execution_results.append(exec_resp)

        if exec_resp.success:
            completed_actions.append(action.action_id)
        else:
            failed_action = action.action_id
            all_successful = False
            execution_errors.extend(exec_resp.errors or [exec_resp.message])
            remaining_actions_aborted = [act.action_id for act in plan.actions[idx + 1:]]

            # If Action N fails after Action 1..N-1 succeeded, dispatch emergency STOP
            if len(completed_actions) > 0:
                logger.warning(
                    f"Action '{failed_action}' failed after actions {completed_actions} succeeded. "
                    "Dispatching emergency STOP to ensure robot safety."
                )
                try:
                    await default_execution_adapter.execute_command(
                        PhysicalExecutionRequest(tool_name="stop", parameters={})
                    )
                except Exception as stop_err:
                    logger.error(f"Emergency stop dispatch error: {stop_err}")

            # Abort execution loop immediately
            break

    # Determine final AssistantStatus
    if all_successful:
        final_status = AssistantStatus.ASSISTANT_EXECUTED
        response_text = plan.plan_explanation or f"Successfully executed {len(completed_actions)} action(s)."
    elif len(completed_actions) > 0:
        final_status = AssistantStatus.ASSISTANT_PARTIAL_FAILURE
        response_text = (
            f"Plan partially executed ({len(completed_actions)} action(s) succeeded). "
            f"Action '{failed_action}' failed; remaining {len(remaining_actions_aborted)} action(s) aborted."
        )
    else:
        final_status = AssistantStatus.ASSISTANT_EXECUTION_FAILED
        response_text = f"Action '{failed_action}' failed to execute: {'; '.join(execution_errors)}"

    # 7. Audit & Memory Persistence
    _log_assistant_audit(
        request_id=request_id,
        session_id=session_id,
        user_message=user_msg,
        plan=plan,
        status_val=final_status,
        completed=completed_actions,
        failed=failed_action,
        aborted=remaining_actions_aborted,
        errors=execution_errors,
    )

    if manager:
        manager.save_assistant_message(session_id, response_text)

    return AssistantResponse(
        request_id=request_id,
        session_id=session_id,
        plan_id=plan.plan_id,
        status=final_status,
        response_text=response_text,
        executed=all_successful,
        plan=plan,
        execution_results=execution_results,
        completed_actions=completed_actions,
        failed_action=failed_action,
        remaining_actions_aborted=remaining_actions_aborted,
        errors=execution_errors,
    )
