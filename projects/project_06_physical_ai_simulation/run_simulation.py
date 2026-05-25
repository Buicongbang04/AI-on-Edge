"""Run the Physical AI loop on the toy 2D grid-world (Chapters 14-16)."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent))  # so we can `from src.physical_ai import ...`

from src.physical_ai import (
    GridWorldPerception, GoalSeekDecision, DiffDriveController, SafetyGate,
)
from world import Robot, World


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    # Build the world + robot
    w_cfg = cfg["world"]
    world = World(size=tuple(w_cfg["size"]), obstacles=[tuple(o) for o in w_cfg["obstacles"]],
                  goal=tuple(w_cfg["goal"]))
    robot = Robot(
        x=float(w_cfg["start"][0]), y=float(w_cfg["start"][1]), theta=float(w_cfg["start"][2]),
        wheel_base=float(cfg["robot"]["wheel_base"]),
    )

    perception = GridWorldPerception(
        noise_std=float(cfg["perception"]["noise_std"]),
        confidence_floor=float(cfg["perception"]["confidence_floor"]),
    )
    decision = GoalSeekDecision(
        goal_tolerance=float(cfg["decision"]["goal_tolerance"]),
        obstacle_safety_dist=float(cfg["decision"]["obstacle_safety_dist"]),
        max_forward=float(cfg["decision"]["max_forward"]),
        turn_gain=float(cfg["decision"]["turn_gain"]),
    )
    controller = DiffDriveController(
        wheel_base=float(cfg["robot"]["wheel_base"]),
        max_wheel_speed=float(cfg["robot"]["max_wheel_speed"]),
    )
    safety = SafetyGate(
        max_wheel_speed=float(cfg["safety"]["max_wheel_speed"]),
        hard_stop_distance=float(cfg["safety"]["hard_stop_distance"]),
        min_confidence=float(cfg["safety"]["min_confidence"]),
        human_override=bool(cfg["safety"]["human_override"]),
    )

    dt = float(cfg["simulation"]["dt"])
    max_steps = int(cfg["simulation"]["max_steps"])
    goal_tol = decision.goal_tolerance

    # Event log + trajectory
    events: list[dict] = []
    trajectory: list[tuple[float, float]] = [(robot.x, robot.y)]
    safety_blocks_by_reason: dict[str, int] = {}
    total_distance = 0.0
    iter_times_ms: list[float] = []
    reached = False

    for step in range(max_steps):
        t0 = time.perf_counter()
        obs = perception.observe(robot.pose, world.goal, world.obstacles)
        action = decision.decide(obs)
        cmd = controller.control(action)
        verdict = safety.gate(cmd, obs)
        prev = (robot.x, robot.y)
        robot.step(verdict.command.v_left, verdict.command.v_right, dt)
        iter_times_ms.append((time.perf_counter() - t0) * 1000.0)

        total_distance += math.hypot(robot.x - prev[0], robot.y - prev[1])
        trajectory.append((robot.x, robot.y))
        if verdict.blocked:
            safety_blocks_by_reason[verdict.reason] = safety_blocks_by_reason.get(verdict.reason, 0) + 1

        events.append({
            "t": round(step * dt, 4),
            "pose": [round(robot.x, 4), round(robot.y, 4), round(robot.theta, 4)],
            "obs": {
                "distance_to_goal": round(obs.distance_to_goal, 4),
                "bearing_to_goal":  round(obs.bearing_to_goal, 4),
                "nearest_obstacle_distance":
                    round(obs.nearest_obstacle_distance, 4) if obs.nearest_obstacle_distance != float("inf") else None,
                "confidence": round(obs.confidence, 4),
            },
            "action": {"forward_speed": round(action.forward_speed, 4),
                       "turn_rate": round(action.turn_rate, 4)},
            "command": {"v_left": round(cmd.v_left, 4),
                        "v_right": round(cmd.v_right, 4)},
            "safety": {"blocked": bool(verdict.blocked),
                       "reason": verdict.reason,
                       "final_command": {"v_left": round(verdict.command.v_left, 4),
                                          "v_right": round(verdict.command.v_right, 4)}},
        })

        if obs.distance_to_goal <= goal_tol:
            reached = True
            break

    # Trajectory plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, world.size[0])
    ax.set_ylim(0, world.size[1])
    ax.set_aspect("equal")
    for ox, oy in world.obstacles:
        ax.add_patch(plt.Circle((ox, oy), decision.obstacle_safety_dist,
                                color="red", alpha=0.15))
        ax.scatter([ox], [oy], c="red", marker="x", s=100, label="_obstacle")
    ax.scatter([world.goal[0]], [world.goal[1]], c="green", marker="*", s=200, label="goal")
    xs, ys = zip(*trajectory)
    ax.plot(xs, ys, color="#1565C0", linewidth=1.5, label="trajectory")
    ax.scatter([trajectory[0][0]], [trajectory[0][1]], c="black", marker="o", s=60, label="start")
    ax.scatter([trajectory[-1][0]], [trajectory[-1][1]], c="purple", marker="s", s=60, label="end")
    ax.set_title(f"Physical AI loop: {'goal reached' if reached else 'did not reach goal'} "
                 f"in {len(events)} steps")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out / "trajectory.png", dpi=120)
    print(f"saved {args.out / 'trajectory.png'}")

    # Event log
    (args.out / "event_log.json").write_text(json.dumps(events, indent=2))
    print(f"wrote {args.out / 'event_log.json'}  ({len(events)} events)")

    # Summary
    import numpy as np
    arr = np.array(iter_times_ms)
    summary = {
        "reached_goal": reached,
        "steps": len(events),
        "total_distance": round(total_distance, 4),
        "safety_blocks_by_reason": safety_blocks_by_reason,
        "final_pose": list(robot.pose),
        "iter_mean_ms": float(arr.mean()),
        "iter_p95_ms": float(np.percentile(arr, 95)),
    }
    (args.out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
