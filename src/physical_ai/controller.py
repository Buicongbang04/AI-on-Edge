"""Controller module — convert a high-level Action into low-level actuator commands.

Two-wheel differential drive:
    v_left  = forward_speed - turn_rate * (wheel_base / 2)
    v_right = forward_speed + turn_rate * (wheel_base / 2)

In a real robot you would also clamp to actuator limits, ramp-rate-limit (so
torque is not torn off violently), and emit motor PWM or torque commands.
"""
from __future__ import annotations

from dataclasses import dataclass

from .decision import Action


@dataclass
class ActuatorCommand:
    """Low-level command sent to the actuator (after safety gate)."""
    v_left: float
    v_right: float


class DiffDriveController:
    """Differential-drive controller with simple clamping."""

    def __init__(self, wheel_base: float = 0.3, max_wheel_speed: float = 1.0) -> None:
        self.wheel_base = float(wheel_base)
        self.max_wheel_speed = float(max_wheel_speed)

    def control(self, action: Action) -> ActuatorCommand:
        half = 0.5 * self.wheel_base
        vl = action.forward_speed - action.turn_rate * half
        vr = action.forward_speed + action.turn_rate * half
        vl = max(-self.max_wheel_speed, min(self.max_wheel_speed, vl))
        vr = max(-self.max_wheel_speed, min(self.max_wheel_speed, vr))
        return ActuatorCommand(v_left=vl, v_right=vr)
