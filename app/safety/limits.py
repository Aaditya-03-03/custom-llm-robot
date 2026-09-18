"""
Hardware safety thresholds, joint angle constraints, and velocity limits.
"""

# Joint angle range mapping: joint_id -> (min_angle_deg, max_angle_deg)
JOINT_LIMITS = {
    0: (0.0, 180.0),    # Base joint
    1: (15.0, 165.0),   # Shoulder joint
    2: (0.0, 150.0),    # Elbow joint
    3: (-90.0, 90.0),   # Wrist pitch
    4: (-180.0, 180.0), # Wrist roll
}

MAX_VELOCITY_DEG_PER_SEC = 90.0
