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
from app.state import default_state_manager, ConnectionStatus
from app.verification.models import (
    ExecutionPhase,
    VerificationStatus,
    RecoveryAction,
    ExecutionRecord,
)
from app.verification.verifier import verify_execution
from app.verification.recovery import determine_recovery_policy, execute_controlled_recovery
from app.verification.manager import default_execution_manager

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
    state_before: Optional[Dict[str, Any]] = None,
    state_after: Optional[Dict[str, Any]] = None,
    execution_records: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Audit logging to MongoDB assistant_audits collection."""
    client = get_mongo_client()
    if client and is_mongodb_available():
        try:
            db_name = getattr(settings, "MONGODB_DATABASE", getattr(settings, "MONGODB_DB_NAME", "custom_llm_robot"))
            db = client[db_name]
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
                "state_before": state_before,
                "state_after": state_after,
                "execution_records": execution_records or [],
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

    # 5. Robot State Freshness & Availability Gate
    state_before = default_state_manager.get_state()
    has_movement = any(action.tool in ("forward", "backward", "left", "right") for action in plan.actions)
    if has_movement and (state_before.connection_status == ConnectionStatus.DISCONNECTED or state_before.is_stale):
        unavail_msg = "The robot connection is unavailable or stale."
        logger.warning(f"Plan rejected due to stale robot state [{request_id}]: {unavail_msg}")
        if manager:
            manager.save_assistant_message(session_id, unavail_msg)
        _log_assistant_audit(
            request_id=request_id,
            session_id=session_id,
            user_message=user_msg,
            plan=plan,
            status_val=AssistantStatus.ASSISTANT_ROBOT_UNAVAILABLE,
            completed=[],
            failed=None,
            aborted=[act.action_id for act in plan.actions],
            errors=[unavail_msg],
            state_before=state_before.model_dump(),
            state_after=state_before.model_dump(),
        )
        return AssistantResponse(
            request_id=request_id,
            session_id=session_id,
            plan_id=plan.plan_id,
            status=AssistantStatus.ASSISTANT_ROBOT_UNAVAILABLE,
            response_text=unavail_msg,
            executed=False,
            plan=plan,
            errors=[unavail_msg],
        )

    # 6. Execution Authority Gate
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

    # 6. Strictly Sequential Closed-Loop Physical Execution & Verification
    # Each action progresses through CREATED -> VALIDATED -> AUTHORIZED -> DISPATCHED -> EXECUTING
    # -> COMPLETED -> VERIFYING -> (VERIFIED | NOT_VERIFIED | FAILED), with controlled recovery on failure.
    execution_results: List[PhysicalExecutionResponse] = []
    completed_actions: List[str] = []
    failed_action: Optional[str] = None
    remaining_actions_aborted: List[str] = []
    execution_errors: List[str] = []
    action_records: List[Dict[str, Any]] = []
    all_successful = True

    for idx, action in enumerate(plan.actions):
        execution_id = f"exec-{uuid.uuid4().hex[:12]}"
        logger.info(
            f"Executing plan action {idx+1}/{len(plan.actions)}: "
            f"action_id='{action.action_id}', execution_id='{execution_id}', tool='{action.tool}', params={action.parameters}"
        )

        # 6a. Register ExecutionRecord & Advance Phases
        await default_execution_manager.create_record(
            execution_id=execution_id,
            request_id=request_id,
            session_id=session_id,
            plan_id=plan.plan_id,
            action_id=action.action_id,
            command=action.tool.upper() if action.tool else "UNKNOWN",
            speed=action.parameters.get("speed"),
            steps=action.parameters.get("steps"),
        )
        await default_execution_manager.advance_phase(execution_id, ExecutionPhase.VALIDATED)
        await default_execution_manager.advance_phase(execution_id, ExecutionPhase.AUTHORIZED)
        await default_execution_manager.advance_phase(execution_id, ExecutionPhase.DISPATCHED)

        state_before_act = default_state_manager.get_state()
        exec_req = PhysicalExecutionRequest(
            tool_name=action.tool,
            parameters=action.parameters,
        )

        # Invariant 3: EXECUTING phase denotes active Stage 6 execution transaction
        await default_execution_manager.advance_phase(execution_id, ExecutionPhase.EXECUTING)

        # 6b. Strictly sequential execution: awaits ACK/result completely
        exec_resp = await default_execution_adapter.execute_command(exec_req)
        execution_results.append(exec_resp)
        await default_execution_manager.record_stage6_result(execution_id, exec_resp)

        if exec_resp.ack_received:
            await default_execution_manager.advance_phase(execution_id, ExecutionPhase.ACKNOWLEDGED)
        if exec_resp.success:
            await default_execution_manager.advance_phase(execution_id, ExecutionPhase.COMPLETED)

        # 6c. Stage 9 Closed-Loop Verification
        await default_execution_manager.advance_phase(execution_id, ExecutionPhase.VERIFYING)
        state_after_act = default_state_manager.get_state()

        rec = default_execution_manager.get_record(execution_id)
        verif_result = verify_execution(
            execution_record=rec,
            stage6_resp=exec_resp,
            state_before=state_before_act,
            state_after=state_after_act,
        )

        # 6d. Check Verification & Execution Outcome
        if verif_result.verification_status == VerificationStatus.FAILED:
            failed_action = action.action_id
            all_successful = False
            execution_errors.extend(exec_resp.errors or [verif_result.reason])
            remaining_actions_aborted = [act.action_id for act in plan.actions[idx + 1:]]

            # Invariant 6 & 7 & 8: Trigger controlled recovery policy (no automatic movement retries)
            rec_action, already_performed = determine_recovery_policy(
                stage6_resp=exec_resp,
                verification_result=verif_result,
                is_stale=state_after_act.is_stale,
            )
            if rec_action == RecoveryAction.STOP:
                if already_performed:
                    rec_msg = "STOP already dispatched by Stage 6 failsafe."
                else:
                    await default_execution_manager.advance_phase(execution_id, ExecutionPhase.RECOVERY)
                    rec_msg = await execute_controlled_recovery(RecoveryAction.STOP)
                await default_execution_manager.record_recovery(execution_id, rec_action, rec_msg)

            await default_execution_manager.complete_verification(
                execution_id=execution_id,
                result=verif_result,
                observed_state=state_after_act.model_dump(),
                final_phase=ExecutionPhase.FAILED,
            )
            action_records.append(default_execution_manager.get_record(execution_id).model_dump())
            break
        else:
            completed_actions.append(action.action_id)
            final_phase = (
                ExecutionPhase.VERIFIED
                if verif_result.verification_status == VerificationStatus.VERIFIED
                else ExecutionPhase.NOT_VERIFIED
            )
            await default_execution_manager.complete_verification(
                execution_id=execution_id,
                result=verif_result,
                observed_state=state_after_act.model_dump(),
                final_phase=final_phase,
            )
            action_records.append(default_execution_manager.get_record(execution_id).model_dump())

    # Determine final AssistantStatus & truthful response text
    if all_successful:
        final_status = AssistantStatus.ASSISTANT_EXECUTED
        # Truthful response: distinguish if physical motion was unverified
        unverified_actions = [
            r for r in action_records
            if r.get("command") in ("FORWARD", "BACKWARD", "LEFT", "RIGHT")
            and r.get("physical_motion_verified") is None
        ]
        if unverified_actions:
            base_msg = plan.plan_explanation or f"Successfully executed {len(completed_actions)} action(s)."
            response_text = f"{base_msg} (Controller acknowledged command(s); physical motion unverified - no sensors)."
        else:
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
    state_after = default_state_manager.get_state()
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
        state_before=state_before.model_dump() if state_before else None,
        state_after=state_after.model_dump(),
        execution_records=action_records,
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
        execution_records=action_records,
    )
