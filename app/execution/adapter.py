"""
Execution Adapter for IOFT Humanoid Robot.
Connects Stage 5 CommandSafetyValidator to physical ESP32 locomotion execution.
Enforces safety validation, execution authority, concurrency control, and audit logging.
"""

import logging
import uuid
from typing import Dict, Any, Optional
from app.schemas.commands import (
    PhysicalExecutionRequest,
    PhysicalExecutionResponse,
    ExecutionStatus,
)
from app.safety.validator import CommandSafetyValidator
from app.execution.protocol import ESP32Protocol
from app.execution.authority import ExecutionAuthorityManager, default_authority_manager
from app.execution.timer import TimedExecutionController, default_execution_controller
from app.execution.logger import ExecutionLogger
from app.execution.errors import (
    AckTimeoutError,
    RobotUnreachableError,
    NetworkError,
    ExecutionBusyError,
    CapabilityDisabledError,
    InvalidPacketError,
)

logger = logging.getLogger("custom_llm_robot.execution.adapter")


class ExecutionAdapter:
    """
    Main execution orchestrator for physical robot locomotion.
    """

    def __init__(
        self,
        authority_manager: Optional[ExecutionAuthorityManager] = None,
        controller: Optional[TimedExecutionController] = None,
    ):
        self.authority_manager = authority_manager or default_authority_manager
        self.controller = controller or default_execution_controller

    async def execute_command(self, request: PhysicalExecutionRequest) -> PhysicalExecutionResponse:
        execution_id = str(uuid.uuid4())
        tool_name = (request.tool_name or "").lower().strip()
        raw_params = request.parameters or {}

        logger.info(f"Execution request [{execution_id}]: tool='{tool_name}', params={raw_params}")

        # 1. Authoritative Safety Validation (Stage 5 Validator)
        val_result = CommandSafetyValidator.validate_command(
            tool_name=tool_name,
            parameters=raw_params,
        )
        if not val_result.valid:
            logger.warning(f"Execution rejected by CommandSafetyValidator: {val_result.message}")
            ExecutionLogger.log_execution(
                execution_id=execution_id,
                tool=tool_name,
                parameters=raw_params,
                status=ExecutionStatus.INVALID_COMMAND,
                hardware_state_affected=False,
                ack_received=False,
                message=val_result.message,
                errors=val_result.errors,
            )
            return PhysicalExecutionResponse(
                success=False,
                status=ExecutionStatus.INVALID_COMMAND,
                execution_id=execution_id,
                tool=tool_name,
                parameters=raw_params,
                packet_sent=None,
                ack_received=False,
                hardware_state_affected=False,
                duration_seconds=None,
                message=f"Validation failed: {val_result.message}",
                errors=val_result.errors,
            )

        # 2. Protocol Wire Format Translation & Physical Verification Gate
        try:
            packet = ESP32Protocol.format_packet(
                tool_name=val_result.tool,
                parameters=val_result.parameters,
            )
        except (CapabilityDisabledError, InvalidPacketError) as e:
            logger.warning(f"Protocol formatting rejected: {e}")
            ExecutionLogger.log_execution(
                execution_id=execution_id,
                tool=val_result.tool,
                parameters=val_result.parameters,
                status=ExecutionStatus.INVALID_COMMAND,
                hardware_state_affected=False,
                ack_received=False,
                message=str(e),
                errors=[str(e)],
            )
            return PhysicalExecutionResponse(
                success=False,
                status=ExecutionStatus.INVALID_COMMAND,
                execution_id=execution_id,
                tool=val_result.tool,
                parameters=val_result.parameters,
                packet_sent=None,
                ack_received=False,
                hardware_state_affected=False,
                duration_seconds=None,
                message=f"Protocol translation rejected: {e}",
                errors=[str(e)],
            )

        # 3. AI Server Execution Authority Gate
        allowed, auth_msg = self.authority_manager.is_execution_allowed(val_result.tool)
        if not allowed:
            logger.warning(f"Execution rejected by Authority Gate: {auth_msg}")
            ExecutionLogger.log_execution(
                execution_id=execution_id,
                tool=val_result.tool,
                parameters=val_result.parameters,
                status=ExecutionStatus.OWNERSHIP_REJECTED,
                hardware_state_affected=False,
                ack_received=False,
                packet_sent=packet,
                message=auth_msg,
                errors=[auth_msg],
            )
            return PhysicalExecutionResponse(
                success=False,
                status=ExecutionStatus.OWNERSHIP_REJECTED,
                execution_id=execution_id,
                tool=val_result.tool,
                parameters=val_result.parameters,
                packet_sent=packet,
                ack_received=False,
                hardware_state_affected=False,
                duration_seconds=None,
                message=auth_msg,
                errors=[auth_msg],
            )

        # 4. Dispatch Execution
        try:
            if val_result.tool == "stop":
                # Emergency STOP path (cancels any running locomotion and dispatches STOP|0|0.00)
                hw_affected, ack_recv, msg = await self.controller.execute_stop()
                status = ExecutionStatus.EXECUTION_SUCCESS if ack_recv else ExecutionStatus.ACK_TIMEOUT
                ExecutionLogger.log_execution(
                    execution_id=execution_id,
                    tool=val_result.tool,
                    parameters=val_result.parameters,
                    status=status,
                    hardware_state_affected=hw_affected,
                    ack_received=ack_recv,
                    packet_sent=packet,
                    duration_seconds=0.0,
                    message=msg,
                )
                return PhysicalExecutionResponse(
                    success=ack_recv,
                    status=status,
                    execution_id=execution_id,
                    tool=val_result.tool,
                    parameters=val_result.parameters,
                    packet_sent=packet,
                    ack_received=ack_recv,
                    hardware_state_affected=hw_affected,
                    duration_seconds=0.0,
                    message=f"STOP executed: {msg}",
                    errors=[] if ack_recv else ["STOP command was not acknowledged by ESP32."],
                )

            else:
                # Locomotion movement path (with concurrency check & timed refresh)
                steps = val_result.parameters.get("steps")
                hw_affected, ack_recv, duration, msg = await self.controller.execute_locomotion(
                    packet=packet,
                    steps=steps,
                )
                ExecutionLogger.log_execution(
                    execution_id=execution_id,
                    tool=val_result.tool,
                    parameters=val_result.parameters,
                    status=ExecutionStatus.EXECUTION_SUCCESS,
                    hardware_state_affected=hw_affected,
                    ack_received=ack_recv,
                    packet_sent=packet,
                    duration_seconds=duration,
                    message=msg,
                )
                return PhysicalExecutionResponse(
                    success=True,
                    status=ExecutionStatus.EXECUTION_SUCCESS,
                    execution_id=execution_id,
                    tool=val_result.tool,
                    parameters=val_result.parameters,
                    packet_sent=packet,
                    ack_received=ack_recv,
                    hardware_state_affected=hw_affected,
                    duration_seconds=duration,
                    message=msg,
                    errors=[],
                )

        except ExecutionBusyError as e:
            logger.warning(f"Locomotion execution rejected (BUSY): {e}")
            ExecutionLogger.log_execution(
                execution_id=execution_id,
                tool=val_result.tool,
                parameters=val_result.parameters,
                status=ExecutionStatus.EXECUTION_BUSY,
                hardware_state_affected=False,
                ack_received=False,
                packet_sent=packet,
                message=str(e),
                errors=[str(e)],
            )
            return PhysicalExecutionResponse(
                success=False,
                status=ExecutionStatus.EXECUTION_BUSY,
                execution_id=execution_id,
                tool=val_result.tool,
                parameters=val_result.parameters,
                packet_sent=packet,
                ack_received=False,
                hardware_state_affected=False,
                duration_seconds=None,
                message=str(e),
                errors=[str(e)],
            )

        except AckTimeoutError as e:
            logger.error(f"ACK Timeout during execution: {e}")
            ExecutionLogger.log_execution(
                execution_id=execution_id,
                tool=val_result.tool,
                parameters=val_result.parameters,
                status=ExecutionStatus.ACK_TIMEOUT,
                hardware_state_affected=True,
                ack_received=False,
                packet_sent=packet,
                message=str(e),
                errors=[str(e)],
            )
            return PhysicalExecutionResponse(
                success=False,
                status=ExecutionStatus.ACK_TIMEOUT,
                execution_id=execution_id,
                tool=val_result.tool,
                parameters=val_result.parameters,
                packet_sent=packet,
                ack_received=False,
                hardware_state_affected=True,
                duration_seconds=None,
                message=f"Execution failed due to ACK timeout: {e}",
                errors=[str(e)],
            )

        except RobotUnreachableError as e:
            logger.error(f"Robot unreachable: {e}")
            ExecutionLogger.log_execution(
                execution_id=execution_id,
                tool=val_result.tool,
                parameters=val_result.parameters,
                status=ExecutionStatus.ROBOT_UNREACHABLE,
                hardware_state_affected=False,
                ack_received=False,
                packet_sent=packet,
                message=str(e),
                errors=[str(e)],
            )
            return PhysicalExecutionResponse(
                success=False,
                status=ExecutionStatus.ROBOT_UNREACHABLE,
                execution_id=execution_id,
                tool=val_result.tool,
                parameters=val_result.parameters,
                packet_sent=packet,
                ack_received=False,
                hardware_state_affected=False,
                duration_seconds=None,
                message=f"Robot host unreachable: {e}",
                errors=[str(e)],
            )

        except NetworkError as e:
            logger.error(f"Network error during execution: {e}")
            ExecutionLogger.log_execution(
                execution_id=execution_id,
                tool=val_result.tool,
                parameters=val_result.parameters,
                status=ExecutionStatus.NETWORK_ERROR,
                hardware_state_affected=False,
                ack_received=False,
                packet_sent=packet,
                message=str(e),
                errors=[str(e)],
            )
            return PhysicalExecutionResponse(
                success=False,
                status=ExecutionStatus.NETWORK_ERROR,
                execution_id=execution_id,
                tool=val_result.tool,
                parameters=val_result.parameters,
                packet_sent=packet,
                ack_received=False,
                hardware_state_affected=False,
                duration_seconds=None,
                message=f"Network communication error: {e}",
                errors=[str(e)],
            )


# Default singleton adapter instance
default_execution_adapter = ExecutionAdapter()
