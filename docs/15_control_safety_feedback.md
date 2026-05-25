# Chapter 15 — Control loop, safety layer, and feedback

> **Goal:** Introduce control and safety fundamentals for Physical AI. By the end of this chapter you should be able to distinguish open-loop and closed-loop control, sketch a basic controller (P / PI / PID concept), insert a safety gate before every action command, and log action/state events for after-the-fact analysis.

This chapter is the "non-AI" half of Physical AI. The control and safety patterns here come from classical control theory and functional safety engineering — they predate modern ML by 50+ years and remain mandatory.

---

## 1. Open-loop vs closed-loop

- **Open-loop**: the controller sends a command and *assumes* the actuator achieves it. No feedback. Cheap, simple, fragile — a stuck motor, a slippery floor, or a load change silently breaks it.
- **Closed-loop**: the controller continuously observes the state and adjusts the command to drive the system toward a target. The whole Physical AI loop (Ch 14) is closed-loop by construction.

| Property | Open-loop | Closed-loop |
|---|---|---|
| Sensors needed | None / minimal | Required |
| Robust to disturbance | No | Yes |
| Cost / complexity | Lower | Higher |
| Stability | Trivial | Must be designed |
| Examples | Toaster timer, garage door | Cruise control, robot arm, drone |

In this course every system is closed-loop. The `src/physical_ai/` modules form a closed loop because the perception observes the result of the previous action.

---

## 2. Controllers: P, PI, PID (concept-level)

Most engineering controllers fall on the same shape:

```
error(t) = target − measurement(t)
control(t) = Kp · error(t)
           + Ki · ∫ error(τ) dτ        ← integral term (PI / PID)
           + Kd · d/dt error(t)         ← derivative term (PID)
```

- **P (proportional)**: scales the response to current error. Always present.
- **I (integral)**: pulls steady-state error to zero. Necessary when there is a constant offset (gravity, friction, bias).
- **D (derivative)**: damps oscillation by reacting to *rate of change* of error.

For learners new to control: a *P-only* controller already handles most teaching tasks. Add I when there is a steady offset; add D only if the system oscillates.

The toy decision module (`GoalSeekDecision`) is a P controller in disguise: `turn = turn_gain × bearing_error` and `forward = clamp(distance × cos(bearing_error))`. Replace it with a proper PID and you get smoother trajectories.

---

## 3. The safety layer

A Physical AI system **must** have a safety layer between the controller and the actuator. The safety layer's job is to *refuse* commands that would violate boundaries.

Minimum safety checks for any robot:

1. **Speed / velocity caps** — clamp wheel/joint speeds to mechanical limits.
2. **Position / joint limits** — refuse motions that exceed the workspace.
3. **Time-to-collision** — predict and abort.
4. **Confidence floor** — if perception confidence is too low, refuse to act.
5. **Battery / temperature interlocks** — refuse to act if power or thermals are unsafe.
6. **Human override** — a switch that always wins; sends "stop" until released.
7. **Heartbeat** — if no command for K ms, fail safe (stop).
8. **Mode interlock** — refuse to drive when in "teaching" mode, refuse to teach when "running", etc.

Industry standards (ISO 13849, ISO 10218 for robots) formalize these. For the course's teaching scope, the safety gate in `src/physical_ai/safety.py` implements checks 1, 3 (distance proxy), 4, and 6.

---

## 4. The course's safety gate

```python
from src.physical_ai import SafetyGate

safety = SafetyGate(
    max_wheel_speed=0.7,        # clamp wheel speeds
    hard_stop_distance=0.25,    # immediate stop if obstacle closer than this
    min_confidence=0.5,         # stop if perception confidence too low
    human_override=False,       # toggle to true to force stop
)

verdict = safety.gate(controller_command, current_observation)
robot.step(verdict.command.v_left, verdict.command.v_right, dt)
log(verdict.reason, verdict.blocked)
```

Every command goes through the gate. The verdict tells the loop:

- **What command actually went to the actuator** (possibly modified).
- **Whether it was blocked or modified** (`verdict.blocked`).
- **Why** (`verdict.reason`) — for logging and operator visibility.

This is a tiny, transparent rule-based gate. Production safety layers add hard-real-time guarantees, redundancy, and certification — but the rule "every command passes through here, and we log what happens" is the same.

---

## 5. Logging actions and failures

Every step in the Physical AI loop should be logged. The course's `project_06_physical_ai_simulation/run_simulation.py` writes per-step events:

```json
{
  "t": 0.45,
  "pose": [1.20, 1.06, 0.39],
  "obs": {"distance_to_goal": 9.41, "bearing_to_goal": 0.71,
          "nearest_obstacle_distance": 3.84, "confidence": 1.0},
  "action": {"forward_speed": 0.62, "turn_rate": 1.06},
  "command": {"v_left": 0.46, "v_right": 0.78},
  "safety": {"blocked": true, "reason": "speed_clamp",
             "final_command": {"v_left": 0.46, "v_right": 0.70}}
}
```

This is the schema you publish over MQTT or write to a per-device log. Chapter 19 (EdgeOps) builds the production version with model versioning, monitoring, and rollback.

---

## 6. Failure-handling patterns

| Failure | Mitigation |
|---|---|
| Sensor disconnect or zero-reading | Treat as `confidence = 0` → safety gate stops the actuator |
| Stale state (last update > timeout) | Heartbeat watchdog → stop |
| Goal unreachable (stuck in loop) | Time-out the goal; surface to operator |
| Latency spike on the perception model | Reuse last good observation for K ms; then stop |
| Power brownout | Persist state; controlled shutdown |
| Operator e-stop | `human_override = true` → all commands become stop |

These patterns belong in the safety gate, the state estimator, or both. They are not the model's problem — they are the *system's* problem.

---

## 7. Timeouts and emergency stop in the simulator

For the toy simulation:

- `max_steps` caps the episode (timeout).
- A simulated e-stop can be triggered by setting `safety.human_override = True` between steps and watching the trajectory.

For a real robot:

- Hardware e-stop button wired to a safety relay that physically removes power from the motors. This is the only true safety guarantee; software safety is *defense in depth*.

---

## 8. The labs

- **`labs/lab_08_robot_simulation_loop.md`** walks through the project_06 simulation step by step: extend the safety gate, induce a failure, and verify the system actually stops.

---

## 9. What you should be able to do after this chapter

- Distinguish open-loop and closed-loop control with an example.
- Explain the role of each term in a PID controller.
- List five safety checks every robot system needs.
- Read the `SafetyGate` code and add a new check (e.g. battery interlock).
- Log every action + safety verdict and produce a per-episode summary.

---

## 10. Files produced by this chapter

- `docs/15_control_safety_feedback.md` — this file.
- `src/physical_ai/safety.py` — example safety gate (extends Ch 14).
- `src/physical_ai/controller.py` — example differential-drive controller (extends Ch 14).
- `labs/lab_08_robot_simulation_loop.md` — guided lab.
