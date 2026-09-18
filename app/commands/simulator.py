"""
Command Simulator for IOFT Humanoid Robot.
Produces canonical validated command representations in dry-run simulation mode.
Zero physical robot execution: no serial, ESP32, GPIO, or motor interaction.
"""

import logging
from typing import Dict, Any, Optional
from app.schemas.commands import DryRunResponse
from app.safety.validator import CommandSafetyValidator

logger = logging.getLogger("custom_llm_robot.commands.simulator")


class CommandSimulator:
    """
    Simulation engine for previewing validated robot command executions.
    Shares the exact same authoritative CommandSafetyValidator pipeline.
    """

    @classmethod
    def simulate_dry_run(
        cls,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> DryRunResponse:
        """
        Execute a dry-run simulation for a robot command.
        Internally calls CommandSafetyValidator to guarantee identical validation rules.
        """
        val_result = CommandSafetyValidator.validate_command(
            tool_name=tool_name,
            parameters=parameters,
        )

        if not val_result.valid:
            logger.warning(
                f"Dry-run simulation rejected invalid command: {tool_name}, reason: {val_result.message}"
            )
            return DryRunResponse(
                success=False,
                execution_mode="simulation",
                tool=val_result.tool,
                parameters=val_result.parameters,
                message=f"Dry-run simulation rejected: {val_result.message}",
                hardware_state_affected=False,
            )

        # Simulation preview: format canonical representation
        logger.info(
            f"Dry-run simulation successful for tool '{val_result.tool}': "
            f"parameters={val_result.parameters}, hardware_state_affected=False"
        )
        return DryRunResponse(
            success=True,
            execution_mode="simulation",
            tool=val_result.tool,
            parameters=val_result.parameters,
            message=(
                f"Command '{val_result.tool}' simulated successfully. "
                "No physical hardware commands dispatched."
            ),
            hardware_state_affected=False,
        )
