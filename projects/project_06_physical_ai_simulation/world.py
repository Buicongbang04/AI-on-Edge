"""Trivial 2D kinematic differential-drive simulator.

No physics — wheel velocities are applied directly to pose with simple kinematic
equations. Adequate to demonstrate the perception → decision → action → feedback
loop without dragging in a physics engine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class World:
    size: tuple[float, float]                  # (width, height)
    obstacles: list[tuple[float, float]]
    goal: tuple[float, float]


@dataclass
class Robot:
    x: float
    y: float
    theta: float
    wheel_base: float = 0.3

    @property
    def pose(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.theta)

    def step(self, v_left: float, v_right: float, dt: float) -> None:
        """Differential-drive kinematics."""
        v = 0.5 * (v_left + v_right)
        omega = (v_right - v_left) / max(self.wheel_base, 1e-6)
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += omega * dt
        # wrap theta to [-pi, pi]
        while self.theta > math.pi:
            self.theta -= 2 * math.pi
        while self.theta < -math.pi:
            self.theta += 2 * math.pi
