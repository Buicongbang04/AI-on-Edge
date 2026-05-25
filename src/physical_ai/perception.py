"""Perception module — turn raw sensor inputs into a structured observation.

For the toy 2D grid-world example used by Chapter 14-16:
- "Sensors" are: own pose (x, y, theta), goal pose, list of obstacles.
- "Perception" computes: distance/bearing to goal, nearest obstacle distance, normalized cues.

In a real robot, this stage replaces the simulator state with model outputs:
camera CNN → object positions, LiDAR → distances, IMU → pose.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Observation:
    """Structured perception output passed to the decision layer."""
    # Robot pose
    x: float
    y: float
    theta: float
    # Goal-relative cues
    distance_to_goal: float
    bearing_to_goal: float          # radians, robot frame: 0 = straight ahead
    # Obstacle cue (closest obstacle; +inf distance, 0 bearing if no obstacle)
    nearest_obstacle_distance: float
    nearest_obstacle_bearing: float = 0.0   # radians from robot heading; 0 = dead ahead
    # Confidence (0..1). Lower means perception is uncertain; safety / decision should use it.
    confidence: float = 1.0


class GridWorldPerception:
    """Simulated perception module for the 2D grid-world.

    In a real robot, the inputs would come from camera + LiDAR + IMU. Here we
    cheat: the simulator gives us ground-truth poses and we compute the
    structured observation directly. Add `noise_std` and `confidence_floor` to
    simulate noisy perception for a more interesting demo.
    """

    def __init__(self, noise_std: float = 0.0, confidence_floor: float = 1.0) -> None:
        self.noise_std = float(noise_std)
        self.confidence_floor = float(confidence_floor)
        self._rng_state = 12345

    def observe(
        self,
        robot_xy_theta: tuple[float, float, float],
        goal_xy: tuple[float, float],
        obstacles_xy: list[tuple[float, float]] | None = None,
    ) -> Observation:
        x, y, theta = robot_xy_theta
        gx, gy = goal_xy
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)

        # Bearing: angle from robot heading to the goal (radians, normalized to [-pi, pi])
        absolute_angle = math.atan2(dy, dx)
        bearing = self._wrap(absolute_angle - theta)

        # Closest obstacle (track both distance and bearing)
        nearest_d = float("inf")
        nearest_b = 0.0
        if obstacles_xy:
            for ox, oy in obstacles_xy:
                d = math.hypot(ox - x, oy - y)
                if d < nearest_d:
                    nearest_d = d
                    nearest_b = self._wrap(math.atan2(oy - y, ox - x) - theta)

        # Apply optional perception noise
        if self.noise_std > 0:
            dist += self._noise() * self.noise_std
            bearing += self._noise() * self.noise_std * 0.1
            if nearest_d != float("inf"):
                nearest_d += self._noise() * self.noise_std

        confidence = max(self.confidence_floor, 1.0 - self.noise_std)
        return Observation(
            x=x, y=y, theta=theta,
            distance_to_goal=max(0.0, dist),
            bearing_to_goal=bearing,
            nearest_obstacle_distance=nearest_d,
            nearest_obstacle_bearing=nearest_b,
            confidence=confidence,
        )

    # --- internals ---
    def _wrap(self, a: float) -> float:
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    def _noise(self) -> float:
        # Tiny LCG so this works without numpy
        self._rng_state = (1103515245 * self._rng_state + 12345) & 0x7FFFFFFF
        return ((self._rng_state / 0x7FFFFFFF) - 0.5) * 2.0
