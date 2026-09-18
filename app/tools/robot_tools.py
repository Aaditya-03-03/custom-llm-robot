"""
Concrete tool implementations for controlling physical or simulated robot actions.
"""

def move_joint(joint_id: int, angle: float) -> bool:
    """Move a specific robot joint to a designated angle."""
    print(f"Moving joint {joint_id} to {angle} degrees")
    return True

def grab_object(gripper_pressure: float = 1.0) -> bool:
    """Actuate end-effector gripper."""
    print(f"Actuating gripper with pressure {gripper_pressure}")
    return True

def stop_robot() -> bool:
    """Trigger emergency stop / halt movement."""
    print("Robot stopped")
    return True
