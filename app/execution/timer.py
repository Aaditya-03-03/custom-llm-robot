"""
Timed Execution Controller for IOFT Humanoid Robot Locomotion.
Enforces single-locomotion concurrency mutual exclusion, sequential non-overlapping
250ms pulse refresh loops, pulse-by-pulse ACK validation, and explicit STOP termination.
"""

import asyncio
import logging
import time
from typing import Tuple, Optional
from app.core.config import settings
from app.execution.client import ESP32Client
from app.execution.errors import ExecutionBusyError, AckTimeoutError

logger = logging.getLogger("custom_llm_robot.execution.timer")


class TimedExecutionController:
    """
    Controls timed locomotion execution and refresh streaming.
    Guarantees mutual exclusion: only one locomotion execution can be active at a time.
    """

    def __init__(self, client: Optional[ESP32Client] = None):
        self.client = client or ESP32Client()
        self._active_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    @property
    def is_busy(self) -> bool:
        """True if an execution coroutine is actively running."""
        return self._active_task is not None and not self._active_task.done()

    async def execute_locomotion(
        self,
        packet: str,
        steps: Optional[int] = None,
    ) -> Tuple[bool, bool, float, str]:
        """
        Execute a locomotion command with timed refresh loop.
        Returns: (hardware_state_affected, ack_received, duration_seconds, message)
        Raises: ExecutionBusyError, AckTimeoutError, NetworkError
        """
        async with self._lock:
            if self.is_busy:
                logger.warning("Locomotion execution rejected: Another command is actively executing.")
                raise ExecutionBusyError("A locomotion command is already executing.")

            step_count = steps if steps is not None else settings.ROBOT_DEFAULT_STEP_COUNT
            total_duration = step_count * settings.ROBOT_STEP_DURATION_SECONDS

            # Create and track the execution task
            current_task = asyncio.current_task()
            self._active_task = current_task

        try:
            logger.info(
                f"Starting timed locomotion execution: packet='{packet}', "
                f"steps={step_count}, total_duration={total_duration:.2f}s"
            )
            start_time = time.monotonic()
            refresh_interval = settings.ROBOT_COMMAND_REFRESH_INTERVAL_SECONDS

            # Sequential pulse loop: strictly one transaction outstanding at a time
            while True:
                now = time.monotonic()
                elapsed = now - start_time
                if elapsed >= total_duration:
                    break

                pulse_start = time.monotonic()
                try:
                    # Send pulse and await MOTOR_ACK (up to 0.5s)
                    await self.client.send_motor_packet(packet)
                except AckTimeoutError as e:
                    logger.error(f"Pulse ACK timed out after {elapsed:.2f}s into execution. Halting robot.")
                    # Immediately dispatch STOP on refresh failure
                    try:
                        await self.client.emergency_stop()
                    except Exception as stop_err:
                        logger.error(f"Failsafe STOP failed after pulse timeout: {stop_err}")
                    raise AckTimeoutError(
                        f"Pulse ACK timeout after {elapsed:.2f}s during timed execution. Emergency STOP dispatched."
                    ) from e

                pulse_duration = time.monotonic() - pulse_start
                remaining_in_step = refresh_interval - pulse_duration
                if remaining_in_step > 0:
                    await asyncio.sleep(remaining_in_step)

            # Normal completion: dispatch explicit STOP packet
            final_elapsed = time.monotonic() - start_time
            logger.info(f"Timed locomotion completed ({final_elapsed:.2f}s). Dispatching explicit STOP...")
            try:
                await self.client.emergency_stop()
            except Exception as e:
                logger.warning(f"Final STOP packet encountered error: {e}")
                raise

            return (
                True,
                True,
                final_elapsed,
                f"Locomotion '{packet}' executed for {final_elapsed:.2f}s and cleanly halted with confirmed STOP.",
            )

        except asyncio.CancelledError:
            logger.info("Locomotion task was cancelled mid-flight (STOP requested).")
            # Failsafe STOP
            try:
                await self.client.emergency_stop()
            except Exception as e:
                logger.error(f"Error dispatching failsafe STOP upon cancellation: {e}")
            raise

        finally:
            async with self._lock:
                if self._active_task is current_task:
                    self._active_task = None

    async def execute_stop(self) -> Tuple[bool, bool, str]:
        """
        Execute emergency STOP:
        1. Cancels active locomotion task if one is running.
        2. Dispatches STOP|0|0.00 and awaits MOTOR_ACK.
        """
        async with self._lock:
            if self.is_busy and self._active_task is not None:
                logger.info("STOP received while locomotion was executing. Cancelling active task...")
                self._active_task.cancel()
                try:
                    # Give cancelled task a moment to cleanly exit
                    await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    pass
                self._active_task = None

        # Dispatch explicit STOP and await ACK
        return await self.client.emergency_stop()


# Singleton controller instance
default_execution_controller = TimedExecutionController()
