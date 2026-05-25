# Project 06 — Physical AI simulation loop

Worked example for Chapters 14-16. A toy 2D grid-world where a differential-drive robot must reach a goal while avoiding obstacles, demonstrating the **perception → state → decision → controller → safety → actuator → environment** loop.

Everything runs in plain Python on a laptop — no robot, no simulator install required.

---

## Files

```
project_06_physical_ai_simulation/
├── README.md                — this file
├── config.yaml              — world / robot / safety parameters
├── world.py                 — tiny 2D simulator (kinematic, no physics)
├── run_simulation.py        — run the full loop, log events, save trajectory plot
└── results/
    ├── trajectory.png
    └── event_log.json
```

---

## Quick start

```bash
cd projects/project_06_physical_ai_simulation
python run_simulation.py
# → results/trajectory.png      (path of the robot from start to goal)
# → results/event_log.json      (every step's observation, action, command, safety verdict)
```

---

## What the simulation does, step by step

The loop:

```python
while not done:
    obs       = perception.observe(robot.pose, goal, obstacles)   # sensors → structured observation
    action    = decision.decide(obs)                              # rule-based policy → high-level action
    cmd       = controller.control(action)                        # action → wheel velocities
    verdict   = safety.gate(cmd, obs)                             # rule-based safety gate
    robot.step(verdict.command, dt)                               # apply to "actuator" (the simulator)
    log_event(obs, action, cmd, verdict)
```

Each component is in `src/physical_ai/`. The loop itself is in `run_simulation.py`. The simulator (`world.py`) is intentionally trivial — a kinematic differential drive that advances the pose given wheel velocities.

---

## Configuration

`config.yaml`:

```yaml
world:
  size: [10.0, 10.0]
  start:  [1.0, 1.0, 0.0]      # x, y, theta (radians)
  goal:   [8.0, 8.0]
  obstacles:
    - [4.0, 4.0]
    - [6.0, 3.0]
    - [3.5, 6.5]

robot:
  wheel_base: 0.3
  max_wheel_speed: 1.0

decision:
  goal_tolerance: 0.25
  obstacle_safety_dist: 0.6
  max_forward: 0.7
  turn_gain: 1.5

safety:
  max_wheel_speed: 0.7
  hard_stop_distance: 0.25
  min_confidence: 0.5
  human_override: false

perception:
  noise_std: 0.0              # 0 = perfect sensors; raise for realism
  confidence_floor: 1.0

simulation:
  dt: 0.05
  max_steps: 600
  seed: 0
```

Things to try:

- Add obstacles in the robot's path. Does it avoid them, or does it crash?
- Raise `noise_std` to 0.3 — does the goal-seek policy still converge?
- Lower `confidence_floor` below `safety.min_confidence` — the robot should refuse to move.
- Set `human_override: true` — the robot should immediately stop and stay stopped.

---

## What "Physical AI" means here, concretely

| Layer | Course module | What it would be in a real robot |
|---|---|---|
| **Sensors** | `world.py` (ground-truth simulator) | Camera, LiDAR, IMU, encoders |
| **Perception** | `src/physical_ai/perception.py` | CNN object detector (Ch 10), pose estimator, SLAM |
| **State** | `Observation` dataclass | World model / state estimator |
| **Decision** | `src/physical_ai/decision.py` | Rule policy, RL policy, VLA model |
| **Controller** | `src/physical_ai/controller.py` | PID, MPC, joint-level controller |
| **Safety gate** | `src/physical_ai/safety.py` | Safety nodelet, ROS2 safety controller, e-stop |
| **Actuator** | `world.py` (advances the pose) | Motors, gripper, joints |
| **Environment** | `world.py` (the 2D grid) | The real world |

The course covers each component at a teaching level. Plug in your own perception (camera + YOLO from Ch 10) and policy to extend the toy into a more realistic system.

---

## Event log

Every step is logged to `results/event_log.json` as a JSON array of:

```json
{
  "t": 0.05,
  "pose": [1.05, 1.00, 0.07],
  "obs": {
    "distance_to_goal": 9.85,
    "bearing_to_goal":  0.78,
    "nearest_obstacle_distance": 4.24,
    "confidence": 1.0
  },
  "action": {"forward_speed": 0.70, "turn_rate": 1.16},
  "command": {"v_left": 0.527, "v_right": 0.873},
  "safety": {"blocked": false, "reason": "speed_clamp",
             "final_command": {"v_left": 0.527, "v_right": 0.700}}
}
```

This is the structure you would publish over MQTT or write to a per-device log in a real edge deployment. Chapter 19 (EdgeOps) builds on this schema.

---

## Reporting

Drop the final state into `results/run_summary.json`:

| Field | Example |
|---|---|
| Reached goal? | true |
| Steps taken | 132 |
| Total distance traveled | 11.4 m |
| Safety blocks triggered | 23 |
| Largest safety reason | speed_clamp (18×), hard_stop_distance (5×) |
| Final pose | (8.04, 8.02, 0.83) |
| Wall-clock per loop iter | 0.4 ms (mean) |
| Wall-clock per loop iter P95 | 0.6 ms |
