"""
Async UDP Transport for IOFT Humanoid Robot.
Binds to an OS-assigned ephemeral local port (never 8889 or 8888).
Communicates with ESP32 UDP motor port (rioft.local:8888).
"""

import asyncio
import logging
import socket
from typing import Tuple, Optional
from app.core.config import settings
from app.execution.errors import NetworkError, RobotUnreachableError, AckTimeoutError

logger = logging.getLogger("custom_llm_robot.execution.transport")


class UDPTransport:
    """
    Asynchronous UDP client transport with ephemeral local port binding.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        self.host = host or settings.ESP32_HOST
        self.port = port or settings.ESP32_PORT
        self.timeout = timeout or settings.ESP32_TIMEOUT_SECONDS

    async def resolve_host(self) -> str:
        """Resolve hostname (rioft.local or IP) to IPv4 address."""
        loop = asyncio.get_running_loop()
        try:
            addr_info = await loop.getaddrinfo(
                self.host,
                self.port,
                family=socket.AF_INET,
                type=socket.SOCK_DGRAM,
            )
            if not addr_info:
                raise RobotUnreachableError(f"Could not resolve host '{self.host}'.")
            target_ip = addr_info[0][4][0]
            return target_ip
        except socket.gaierror as e:
            logger.error(f"DNS resolution failed for '{self.host}': {e}")
            raise RobotUnreachableError(f"Robot host '{self.host}' unreachable: DNS resolution failed.") from e
        except Exception as e:
            logger.error(f"Error resolving host '{self.host}': {e}")
            raise RobotUnreachableError(f"Robot host '{self.host}' unreachable: {e}") from e

    async def send_and_receive(
        self,
        packet: str,
        custom_timeout: Optional[float] = None,
    ) -> Tuple[bool, bytes, bool]:
        """
        Send a UDP packet and wait for response.
        Returns: (packet_transmitted, response_bytes, ack_received)
        packet_transmitted is True as soon as sock.sendto succeeds (hardware_state_affected=True).
        """
        timeout = custom_timeout or self.timeout
        target_ip = await self.resolve_host()
        data = packet.encode("utf-8")
        packet_transmitted = False

        loop = asyncio.get_running_loop()

        def _do_udp_transaction() -> Tuple[bool, bytes, bool]:
            nonlocal packet_transmitted
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                # Ephemeral local port assignment (port 0)
                sock.bind(("", 0))
                sock.settimeout(timeout)
                local_port = sock.getsockname()[1]
                logger.debug(f"Bound to ephemeral local UDP port: {local_port}")

                try:
                    sock.sendto(data, (target_ip, self.port))
                    packet_transmitted = True
                    logger.debug(f"Transmitted '{packet}' to {target_ip}:{self.port}")
                except Exception as send_err:
                    logger.error(f"Socket send failed for {target_ip}:{self.port}: {send_err}")
                    raise NetworkError(f"Failed to transmit UDP packet to {target_ip}:{self.port}: {send_err}") from send_err

                try:
                    response, sender_addr = sock.recvfrom(1024)
                    logger.debug(f"Received response from {sender_addr}: {response!r}")
                    return packet_transmitted, response, True
                except socket.timeout as t_err:
                    logger.warning(
                        f"Timeout ({timeout}s) waiting for response from {target_ip}:{self.port} for packet '{packet}'"
                    )
                    raise AckTimeoutError(
                        f"No ACK received from ESP32 at {target_ip}:{self.port} within {timeout}s."
                    ) from t_err
                except Exception as recv_err:
                    logger.error(f"Socket recv error from {target_ip}:{self.port}: {recv_err}")
                    raise NetworkError(f"Failed to receive UDP response from {target_ip}:{self.port}: {recv_err}") from recv_err
            finally:
                sock.close()

        try:
            return await loop.run_in_executor(None, _do_udp_transaction)
        except (AckTimeoutError, NetworkError, RobotUnreachableError):
            raise
        except Exception as e:
            raise NetworkError(f"Unexpected UDP transaction error: {e}") from e
