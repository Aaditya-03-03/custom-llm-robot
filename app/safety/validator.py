"""
Safety validator to ensure LLM generated commands are within safe operational bounds.
"""

from app.safety.limits import JOINT_LIMITS

class CommandSafetyValidator:
    @staticmethod
    def validate_joint_movement(joint_id: int, angle: float) -> tuple[bool, str]:
        """Validate if target joint angle is within safety bounds."""
        if joint_id not in JOINT_LIMITS:
            return False, f"Unknown joint ID {joint_id}"
        min_angle, max_angle = JOINT_LIMITS[joint_id]
        if not (min_angle <= angle <= max_angle):
            return False, f"Angle {angle} out of bounds [{min_angle}, {max_angle}] for joint {joint_id}"
        return True, "Valid"
