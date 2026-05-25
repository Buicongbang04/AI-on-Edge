# Chapter 14 — Physical AI: perception → state → decision → action

> **Goal:** Frame Physical AI as a closed action loop. By the end of this chapter you should be able to describe the perception → state → decision → controller → actuator → feedback pipeline, distinguish perception-only AI from Physical AI, and recognize the role each module plays.

This is the first chapter of **Part 4**. Up to now the course has built systems that *perceive*: classify, detect, segment, transcribe. From here the system must also *decide* and *act* in the physical world — and feed back the result. That changes the design rules.

---

## 1. The loop

```
sensors ─► perception ─► state ─► decision ─► controller ─► safety gate ─► actuator ─► environment
                                                                                            │
                                                                                            ▼
                                                                                    (new sensor reading)
                                                                                            │
                                                                                            └──► sensors ...
```

The six required stages are:

1. **Sensors:** the physical input — cameras, LiDAR, microphones, IMU, encoders, force sensors.
2. **Perception:** turn raw sensor data into a *structured world description*. CNN detector outputs (Ch 10), SLAM map, pose estimator, sensor fusion.
3. **State:** the system's belief about itself + the world — pose, velocity, mode, battery, recent history. Often produced by an estimator (Kalman filter, particle filter, learned state predictor).
4. **Decision:** rule-based logic, classical planner, RL policy, or a vision-language-action (VLA) model. Output is a high-level action.
5. **Controller:** convert the high-level action into low-level commands the actuator can execute — joint torques, wheel velocities, gripper force, motor PWM.
6. **Actuator:** the physical output — wheels, arm joints, gripper, valves, speaker.
7. **Safety gate:** sits between the controller and the actuator. Inspects every command, blocks anything that violates safety (speed, position, force, confidence, mode, override). **Always required.** Ch 15 develops this.

The loop closes when the next sensor reading reflects the physical effect of the action.

---

## 2. Edge AI vs Physical AI

| Property | Edge AI (perception only) | Physical AI (closed loop) |
|---|---|---|
| Input | Sensor stream | Sensor stream |
| Output | Label, box, alert | **Action that changes the world** |
| Feedback | None (open loop) | Yes (closed loop) |
| Failure mode | Bad prediction | Bad prediction + physical consequence |
| Safety required | Recommended | **Mandatory** |
| Latency budget | Often ms-class | Often ms-class with strict P99 |
| Testing | Held-out test set | Simulation + real-world rollout |

A YOLO that draws boxes on a camera feed is Edge AI. A robot that *moves* based on those boxes is Physical AI. Almost all classical robotics is Physical AI, even when it does not call itself "AI".

---

## 3. The five canonical Physical AI systems

| System | Sensors | Decision | Actuator |
|---|---|---|---|
| Autonomous mobile robot (warehouse) | LiDAR + cameras + IMU | Path planner + obstacle avoidance | Wheels |
| Robot arm pick-and-place | Camera + joint encoders + force | Vision detector + motion planner | Joint motors + gripper |
| Drone | IMU + camera + GPS | Attitude controller + visual servoing | Rotor PWM |
| Autonomous vehicle | Camera + LiDAR + radar + GPS | VLA model + behaviour planner | Steering / brake / throttle |
| Industrial inspector with reject arm | Camera + sensor | Defect detector + threshold | Pneumatic rejector |

The course's worked example is the simplest one (a 2D differential-drive robot in a grid world), but the same loop generalizes to all of the above.

---

## 4. Perception is the hard half (usually)

For a Physical AI system on a budget:

- **Perception** drives accuracy, reliability, and most failures.
- **Decision** is often surprisingly simple: a small rule book, a PID, a Bayesian planner. Learned policies (RL) get more attention than they deserve relative to careful perception.
- **Controller** is well-understood from classical control theory.
- **Safety** is often the difference between "demo works" and "deployable".

The course mirrors this: Ch 14-17 spend most of the effort on the *structure of the loop* and the *safety layer*, not on cutting-edge VLA models.

---

## 5. The code interface

`src/physical_ai/` ships four tiny modules that demonstrate one swappable interface per stage:

```python
from src.physical_ai import (
    GridWorldPerception,   # observe(...) -> Observation
    GoalSeekDecision,      # decide(Observation) -> Action
    DiffDriveController,   # control(Action) -> ActuatorCommand
    SafetyGate,            # gate(ActuatorCommand, Observation) -> SafetyVerdict
)
```

The loop is:

```python
obs     = perception.observe(robot.pose, goal, obstacles)
action  = decision.decide(obs)
cmd     = controller.control(action)
verdict = safety.gate(cmd, obs)
robot.step(verdict.command.v_left, verdict.command.v_right, dt)
```

Each stage is independently testable. To swap in a learned policy, replace `GoalSeekDecision` with one that consumes the same `Observation` and returns the same `Action`.

---

## 6. World state vs perception output

A subtle distinction: the **perception output** is what the model can *measure*. The **state** is what the system *believes* about itself and the world, which may include:

- Smoothed pose from a Kalman filter (not the raw camera pose).
- Map of the environment (not just the current frame).
- Recent history (last K seconds of observations, useful for momentum, occlusion handling).
- Mode flags: "approaching goal", "avoiding obstacle", "low battery", "operator paused".

In the toy example the state is the `Observation` dataclass — perception output is taken as the state. In real systems, **add an explicit state estimator** before the decision layer.

---

## 7. What you should be able to do after this chapter

- Draw the perception → state → decision → controller → safety → actuator → feedback loop from memory and describe each stage in one sentence.
- Differentiate Edge AI (perception only) from Physical AI (closed loop).
- Read the `src/physical_ai/` modules and identify what each one would look like for a real robot.
- Articulate why safety is mandatory in Physical AI.

---

## 8. Files produced by this chapter

- `docs/14_physical_ai_loop.md` — this file.
- `src/physical_ai/perception.py` — example perception module.
- `src/physical_ai/decision.py` — example rule-based decision policy.
- `src/physical_ai/controller.py` — example diff-drive controller.
- `src/physical_ai/__init__.py` — module exports.

(Chapter 15 adds `src/physical_ai/safety.py`.)
