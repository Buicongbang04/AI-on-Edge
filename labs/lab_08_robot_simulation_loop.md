# Lab 08 — Physical AI simulation loop

**Chapter:** 14-15
**Prerequisites:** finished Chapters 14 and 15. Ran `projects/project_06_physical_ai_simulation/run_simulation.py` once successfully.
**Estimated effort:** 2-3 hours.

The point of this lab is to experience what happens when you change one stage of the perception → decision → controller → safety loop. You will alter parameters, watch the trajectory change, and confirm the safety gate behaves as expected.

---

## Part 1 — Baseline (15 min)

1. From the project directory, run the baseline:

   ```bash
   cd projects/project_06_physical_ai_simulation
   python run_simulation.py
   ```

2. Confirm:
   - `results/trajectory.png` shows a curved path from start (1,1) to goal (8,8) around the obstacles.
   - `results/run_summary.json` reports `"reached_goal": true`.
   - `results/event_log.json` contains 300+ events.

3. Write down the baseline:
   - Steps to goal: ___
   - Total distance: ___
   - Safety blocks by reason: `{ ... }`

---

## Part 2 — Tighten the safety gate (30 min)

Edit `config.yaml`:

```yaml
safety:
  max_wheel_speed: 0.35       # was 0.7 — half the wheel speed
  hard_stop_distance: 0.45    # was 0.25 — wider safety bubble
```

Re-run. Expect:

- More steps to goal (slower robot).
- More frequent `speed_clamp` blocks.
- Possibly the robot gets stuck near an obstacle if `hard_stop_distance > decision.obstacle_safety_dist`.

Record:
- Did it reach the goal? Yes / No
- Steps: ___
- Safety blocks: ___

---

## Part 3 — Force a human override (15 min)

Edit `config.yaml`:

```yaml
safety:
  human_override: true
```

Re-run. Expect:

- Every command is replaced with `(0, 0)`.
- Robot does not move.
- `safety_blocks_by_reason` is dominated by `"human_override"`.

This confirms the override path. Reset to `false` for the next part.

---

## Part 4 — Simulate noisy perception (30 min)

Edit `config.yaml`:

```yaml
perception:
  noise_std: 0.3
  confidence_floor: 0.6
```

Re-run. Expect:

- The trajectory looks noisier — the robot wiggles.
- Possibly the robot still reaches the goal (perception noise is bounded).

Now push it further:

```yaml
perception:
  noise_std: 0.6
  confidence_floor: 0.4         # below safety.min_confidence (0.5)
```

Re-run. Expect:

- The safety gate refuses to act because confidence is too low (`low_confidence(...)`).
- The robot does not move.

This confirms the perception-confidence safety check. Reset.

---

## Part 5 — Block one obstacle path (30 min)

Add a new obstacle right on the goal-direction line:

```yaml
world:
  obstacles:
    - [4.0, 4.0]
    - [6.0, 3.0]
    - [3.5, 6.5]
    - [5.5, 5.5]      # new — sits between start and goal
```

Re-run. The robot should curve around all four obstacles.

If the robot now gets stuck, the local rule-based policy is not enough. Discuss in your report: what would you do? (Hint: a real system would use a planner — A* / RRT — instead of pure reactive avoidance.)

---

## Part 6 — Report (30 min)

Submit `experiments/reports/lab_08_<your_name>.md` with:

```markdown
# Lab 08 report

## Hardware
(this lab runs on laptop CPU; record the OS)

## Baseline
- Steps to goal: ...
- Total distance: ...
- Safety blocks: ...

## Tightened safety
- Reached goal? ...
- Trade-off: ...

## Human override
- Did the robot stop immediately? ...

## Noisy perception
- noise_std=0.3: ...
- noise_std=0.6, confidence_floor=0.4: ...

## New obstacle
- Outcome: ...
- What would a *planner* do that reactive rules cannot? (≤100 words)

## Take-aways (≤200 words)
- One sentence on how perception affected the system.
- One sentence on how the safety gate changed system behavior.
- One sentence on what you would change for a real robot deployment.
```

---

## Grading rubric (100 points)

| Criterion | Points |
|---|---|
| Ran the baseline successfully | 10 |
| Tightened safety; observed the trade-off | 20 |
| Triggered human override; confirmed stop | 15 |
| Triggered low-confidence safety stop | 15 |
| Added a new obstacle and re-ran | 15 |
| Report has all 6 sections, with concrete numbers | 20 |
| Take-aways are specific (not generic) | 5 |
| **Total** | **100** |

---

## Common pitfalls

- Forgetting to re-run after editing `config.yaml`.
- Setting `obstacle_safety_dist < hard_stop_distance` and getting stuck in the safety bubble. Keep them ordered: `hard_stop_distance < decision.obstacle_safety_dist`.
- Confusing the *decision* obstacle distance with the *safety* hard-stop distance. They serve different purposes (smooth avoidance vs hard stop).
- Reading only the final image — also open `run_summary.json` for the numerical truth.
