"""
Exceptions for Stage 6 Physical Robot Execution Layer.
"""


class ExecutionError(Exception):
    """Base exception for robot execution errors."""
    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.message = message
        self.errors = errors or []


class AckTimeoutError(ExecutionError):
    """Raised when ESP32 does not respond with MOTOR_ACK within the configured timeout."""
    pass


class NetworkError(ExecutionError):
    """Raised on socket/transport network communication failures."""
    pass


class RobotUnreachableError(ExecutionError):
    """Raised when the robot host (rioft.local) cannot be resolved or reached."""
    pass


class CommandRejectedError(ExecutionError):
    """Raised when safety validation rejects the command."""
    pass


class AuthorityConflictError(ExecutionError):
    """Raised when AI server execution authority is MANUAL and AI execution is attempted."""
    pass


class ExecutionBusyError(ExecutionError):
    """Raised when another locomotion execution is already running."""
    pass


class InvalidPacketError(ExecutionError):
    """Raised when protocol encoding or decoding fails."""
    pass


class CapabilityDisabledError(ExecutionError):
    """Raised when capability is physically unverified and disabled (e.g. pivot turns)."""
    pass
