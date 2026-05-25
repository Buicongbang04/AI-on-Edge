"""Safety layer — gate the actuator command on rule-based safety boundaries.

Rules used here:
- Hard cap on wheel speed (`max_wheel_speed`).
- Block all forward motion if perception confidence is below `min_confidence`
  (force stop and surface to a human).
- Block all forward motion if nearest obstacle is below `hard_stop_distance`.
- Optional `human_override` flag immediately commands stop.

This is the minimum viable safety layer for a teaching project. In a real
deployment you would add: kinematic limits, joint limits, time-to-collision
estimates, battery / temperature interlocks, e-stop integration, ROS2 safety
nodelets, MISRA-compliant implementations, etc.
"""
from __future__ import annotations

from dataclasses import dataclass

from .controller import ActuatorCommand
from .perception import Observation


@dataclass
class SafetyVerdict:
    """Result of the safety gate."""
    command: ActuatorCommand     # the (possibly modified) command actually emitted
    blocked: bool                # True if the original command was modified or blocked
    reason: str                  # short, human-readable description of why


class SafetyGate:
    def __init__(
        self,
        max_wheel_speed: float = 0.7,
        hard_stop_distance: float = 0.25,
        min_confidence: float = 0.5,
        human_override: bool = False,
    ) -> None:
        self.max_wheel_speed = float(max_wheel_speed)
        self.hard_stop_distance = float(hard_stop_distance)
        self.min_confidence = float(min_confidence)
        self.human_override = bool(human_override)

    def gate(self, cmd: ActuatorCommand, obs: Observation) -> SafetyVerdict:
        # 1. Human override always wins
        if self.human_override:
            return SafetyVerdict(ActuatorCommand(0.0, 0.0), True, "human_override")

        # 2. Low perception confidence → stop and surface to operator
        if obs.confidence < self.min_confidence:
            return SafetyVerdict(ActuatorCommand(0.0, 0.0), True,
                                 f"low_confidence({obs.confidence:.2f})")

        # 3. Hard obstacle stop
        if obs.nearest_obstacle_distance < self.hard_stop_distance:
            return SafetyVerdict(ActuatorCommand(0.0, 0.0), True,
                                 f"hard_stop_distance({obs.nearest_obstacle_distance:.2f})")

        # 4. Speed cap
        vl = max(-self.max_wheel_speed, min(self.max_wheel_speed, cmd.v_left))
        vr = max(-self.max_wheel_speed, min(self.max_wheel_speed, cmd.v_right))
        clamped = (vl != cmd.v_left) or (vr != cmd.v_right)
        return SafetyVerdict(ActuatorCommand(vl, vr), clamped,
                             "speed_clamp" if clamped else "ok")
