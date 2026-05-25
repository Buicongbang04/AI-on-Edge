# Lab 09 — Mock ROS-style pub/sub for the Physical AI loop

**Chapter:** 17
**Prerequisites:** finished `project_06_physical_ai_simulation` (Ch 14-16).
**Estimated effort:** 2 hours.

The goal of this lab is to wrap the toy Physical AI loop in a **publisher / subscriber** architecture that mimics ROS2 — without installing ROS2. Once the toy works this way, switching to real ROS2 later is mostly a syntactic change.

---

## Part 1 — The mock bus (30 min)

Create `experiments/lab_09/mock_bus.py`:

```python
class Bus:
    """A tiny synchronous pub/sub bus."""

    def __init__(self) -> None:
        self._subs: dict[str, list] = {}

    def subscribe(self, topic: str, cb) -> None:
        self._subs.setdefault(topic, []).append(cb)

    def publish(self, topic: str, msg) -> None:
        for cb in self._subs.get(topic, []):
            cb(msg)
```

This is not asynchronous, not multi-process — but it has the right *shape*. In ROS2 each callback runs in its node's executor; here each callback runs synchronously when you publish. Good enough to learn the pattern.

---

## Part 2 — Wrap each Physical AI module as a "node" (60 min)

Each module subscribes to its inputs and publishes its outputs. Reuse the existing `src/physical_ai/` modules — only the I/O changes.

Create `experiments/lab_09/run_mock_ros.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

from src.physical_ai import (
    GridWorldPerception, GoalSeekDecision, DiffDriveController, SafetyGate,
)
from projects.project_06_physical_ai_simulation.world import Robot, World
from mock_bus import Bus


def main() -> None:
    bus = Bus()
    world = World(size=(10.0, 10.0),
                  obstacles=[(4.0, 4.0), (6.0, 3.0), (3.5, 6.5)],
                  goal=(8.0, 8.0))
    robot = Robot(1.0, 1.0, 0.0)

    perception = GridWorldPerception()
    decision   = GoalSeekDecision(obstacle_safety_dist=0.6, max_forward=0.7, turn_gain=1.5)
    controller = DiffDriveController(wheel_base=0.3, max_wheel_speed=1.0)
    safety     = SafetyGate(max_wheel_speed=0.7, hard_stop_distance=0.25, min_confidence=0.5)

    # --- Decision node: subscribes /observation, publishes /cmd_high ---
    def on_observation(obs):
        action = decision.decide(obs)
        bus.publish("/cmd_high", action)

    # --- Controller node: subscribes /cmd_high, publishes /wheel_cmd ---
    last_obs = {"v": None}
    def on_cmd_high(action):
        cmd = controller.control(action)
        bus.publish("/wheel_cmd", (cmd, last_obs["v"]))

    # --- Safety node: subscribes /wheel_cmd, publishes /wheel_cmd_safe ---
    def on_wheel_cmd(pair):
        cmd, obs = pair
        verdict = safety.gate(cmd, obs)
        bus.publish("/wheel_cmd_safe", verdict)

    # --- Sim node: subscribes /wheel_cmd_safe, advances pose, publishes /odom ---
    state = {"reached": False, "steps": 0}
    def on_wheel_safe(verdict):
        robot.step(verdict.command.v_left, verdict.command.v_right, dt=0.05)
        state["steps"] += 1
        bus.publish("/odom", (robot.x, robot.y, robot.theta))

    bus.subscribe("/observation", on_observation)
    bus.subscribe("/cmd_high", on_cmd_high)
    bus.subscribe("/wheel_cmd", on_wheel_cmd)
    bus.subscribe("/wheel_cmd_safe", on_wheel_safe)

    # --- "Sensor" loop driver: publishes /observation each tick ---
    for _ in range(600):
        obs = perception.observe(robot.pose, world.goal, world.obstacles)
        last_obs["v"] = obs
        bus.publish("/observation", obs)
        if obs.distance_to_goal < decision.goal_tolerance:
            state["reached"] = True
            break

    print(f'reached={state["reached"]}  steps={state["steps"]}  '
          f'final_pose={robot.pose}')


if __name__ == "__main__":
    main()
```

Run it:

```bash
mkdir -p experiments/lab_09
# (paste mock_bus.py and run_mock_ros.py as above)
python experiments/lab_09/run_mock_ros.py
```

Expected output:

```
reached=True  steps=~370  final_pose=(~8, ~8, ~)
```

---

## Part 3 — Add a logger node (15 min)

A logger node subscribes to `/wheel_cmd_safe` and `/odom` and writes a JSONL file. This mimics `rosbag2`.

Add to your script:

```python
import json
log = open("experiments/lab_09/log.jsonl", "w")
def on_log_odom(p):
    log.write(json.dumps({"topic": "/odom", "x": p[0], "y": p[1], "theta": p[2]}) + "\n")
def on_log_safe(v):
    log.write(json.dumps({"topic": "/wheel_cmd_safe", "v_left": v.command.v_left,
                          "v_right": v.command.v_right, "blocked": v.blocked,
                          "reason": v.reason}) + "\n")
bus.subscribe("/odom", on_log_odom)
bus.subscribe("/wheel_cmd_safe", on_log_safe)
```

Re-run and inspect `log.jsonl`. You now have a per-topic event log that any other node (a visualizer, a metrics aggregator) could consume.

---

## Part 4 — Report (15 min)

Submit `experiments/reports/lab_09_<your_name>.md` with:

```markdown
# Lab 09 report

## Topology
- Topics created: ...
- Nodes (subscribers + publishers): ...

## Behavior
- Did the robot still reach the goal? ...
- How many steps? ...

## Comparing the mock to real ROS2
- One thing this mock does NOT capture about real ROS2: ...
- One reason you would use ROS2 for a real robot: ...

## Take-aways (≤200 words)
- Why this architecture is easier to test than a monolithic loop.
- Where you would put a learned policy in this graph (which topic does it subscribe to / publish on?).
```

---

## Grading rubric (100 points)

| Criterion | Points |
|---|---|
| Mock bus implemented | 15 |
| All four "nodes" wired via topics | 25 |
| Robot still reaches the goal | 20 |
| Logger node writes a JSONL log | 15 |
| Report sections all answered | 20 |
| Take-aways are specific | 5 |
| **Total** | **100** |

---

## Common pitfalls

- Topic name typos: in a real ROS2 setup, a typo means silent no-op. The mock bus is the same. Print topic names.
- Forgetting to make the safety node subscribe BEFORE the controller publishes — order matters in a synchronous bus.
- The mock is single-threaded; in real ROS2 each callback runs in an executor. Be aware before you assume the patterns transfer 1:1 for timing-sensitive code.
