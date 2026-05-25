"""Physical AI building blocks (Chapters 14-17).

Modules:
    perception  — turn sensor inputs into a structured observation
    decision    — rule-based or learned policy that picks an action
    controller  — convert action into low-level commands (velocity, PWM, torque)
    safety      — safety gate that filters action commands before they reach the actuator

Glue them in this order:
    sensors -> perception -> state -> decision -> controller -> safety -> actuator -> environment
                                                                                       |
                                              feedback (new sensor readings) <---------+
"""
from .perception import GridWorldPerception, Observation
from .decision import GoalSeekDecision, Action
from .controller import DiffDriveController, ActuatorCommand
from .safety import SafetyGate, SafetyVerdict

__all__ = [
    "GridWorldPerception", "Observation",
    "GoalSeekDecision", "Action",
    "DiffDriveController", "ActuatorCommand",
    "SafetyGate", "SafetyVerdict",
]
