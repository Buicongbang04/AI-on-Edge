"""Decision module — rule-based policy that picks an action from an observation.

For Chapter 14-16's toy: a simple goal-seeking policy that turns toward the
goal and moves forward at a speed that scales with distance. If the closest
obstacle is too near, it backs off and rotates.

This is the place where, in a real Physical AI system, you would plug in:
- a learned policy (RL, imitation),
- a vision-language-action model,
- or a higher-level planner.

The interface is intentionally tiny so all three can substitute cleanly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .perception import Observation


@dataclass
class Action:
    """High-level action — semantic, not actuator-specific.

    `forward_speed`  in [-1, 1] (negative = reverse)
    `turn_rate`      in [-1, 1] (negative = right turn)
    """
    forward_speed: float
    turn_rate: float


class GoalSeekDecision:
    """Rule-based goal-seeking policy.

    Behaviour:
    - If close to an obstacle: back off and rotate away.
    - Otherwise: turn toward the goal, move forward at speed ∝ distance (capped).

    Parameters:
        goal_tolerance       : stop forward motion within this distance.
        obstacle_safety_dist : trigger avoidance below this distance.
        max_forward          : top forward speed in [0, 1].
        turn_gain            : how aggressively to turn toward the goal.
    """

    def __init__(
        self,
        goal_tolerance: float = 0.2,
        obstacle_safety_dist: float = 0.5,
        max_forward: float = 1.0,
        turn_gain: float = 1.5,
    ) -> None:
        self.goal_tolerance = float(goal_tolerance)
        self.obstacle_safety_dist = float(obstacle_safety_dist)
        self.max_forward = float(max_forward)
        self.turn_gain = float(turn_gain)

    def decide(self, obs: Observation) -> Action:
        # Obstacle avoidance takes precedence
        if obs.nearest_obstacle_distance < self.obstacle_safety_dist:
            return Action(forward_speed=-0.3, turn_rate=0.8)

        if obs.distance_to_goal <= self.goal_tolerance:
            return Action(forward_speed=0.0, turn_rate=0.0)

        # Forward speed scales with distance (capped at max_forward).
        # We slow down when bearing error is large (avoid swinging far off course).
        bearing_factor = max(0.0, math.cos(obs.bearing_to_goal))
        forward = min(self.max_forward, obs.distance_to_goal) * bearing_factor

        # Turn proportional to bearing
        turn = max(-1.0, min(1.0, self.turn_gain * obs.bearing_to_goal))
        return Action(forward_speed=forward, turn_rate=turn)
