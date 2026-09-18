"""
Pydantic schemas for robot hardware state and telemetry models.
"""

from pydantic import BaseModel, Field

class JointState(BaseModel):
    joint_id: int
    angle: float
    velocity: float = 0.0

class RobotState(BaseModel):
    is_connected: bool = False
    e_stop_active: bool = False
    joints: list[JointState] = Field(default_factory=list)
