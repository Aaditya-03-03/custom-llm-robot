"""
High-level ESP32 Robot Hardware Client for IOFT Humanoid Robot.
Manages packet transmission, ACK validation, and emergency stop without blind retries.
"""

import logging
from typing import Tuple, Optional
from app.execution.transport import UDPTransport
from app.execution.protocol import ESP32Protocol
from app.execution.errors import AckTimeoutError, NetworkError, RobotUnreachableError

logger = logging.getLogger("custom_llm_robot.execution.client")


class ESP32Client:
    """
    Client for interacting with the ESP32 locomotion controller over UDP.
    """

    def __init__(self, transport: Optional[UDPTransport] = None):
        self.transport = transport or UDPTransport()

    async def send_motor_packet(self, packet: str) -> Tuple[bool, bool, str]:
        """
        Send a motor command packet and wait for MOTOR_ACK.
        Returns: (hardware_state_affected, ack_received, message)
        """
        try:
            packet_transmitted, response_bytes, _ = await self.transport.send_and_receive(packet)
            if ESP32Protocol.is_valid_ack(response_bytes):
                logger.info(f"Packet '{packet}' acknowledged by ESP32 (MOTOR_ACK).")
                return True, True, "MOTOR_ACK received"
            else:
                logger.warning(f"Unexpected response from ESP32: {response_bytes!r}")
                return True, False, f"Unexpected response from ESP32: {response_bytes!r}"

        except AckTimeoutError as e:
            logger.warning(f"ACK timeout for packet '{packet}': {e}")
            # Hardware state affected is True because packet was transmitted to socket
            raise
        except (RobotUnreachableError, NetworkError) as e:
            logger.error(f"Network error transmitting packet '{packet}': {e}")
            raise

    async def emergency_stop(self) -> Tuple[bool, bool, str]:
        """
        High-priority emergency stop dispatch: sends STOP|0|0.00 and awaits MOTOR_ACK.
        Returns: (hardware_state_affected, ack_received, message)
        """
        stop_packet = ESP32Protocol.format_packet("stop")
        logger.info(f"Dispatching emergency stop packet: '{stop_packet}'")
        return await self.send_motor_packet(stop_packet)
