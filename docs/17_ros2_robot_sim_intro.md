# Chapter 17 — ROS2 / robot simulator intro

> **Goal:** Bridge from this course's Physical AI fundamentals to robotics engineering. By the end of this chapter you should know the ROS2 concepts (node, topic, message, service), understand the publisher / subscriber pattern, and be able to build a *mock* ROS-like pub/sub in Python so the toy loop in `project_06` can be expressed in robotics-style architecture even when ROS2 is not installed.

ROS2 is *the* open-source robotics middleware in 2026. Almost every research and industrial robot stack speaks it, including the NVIDIA Isaac stack, most ROS-based robot arms, drones (PX4/ROS bridge), and SLAM stacks. This chapter is **orientation, not installation** — the course does not require ROS2.

---

## 1. What ROS2 is

ROS2 is **middleware**: a set of libraries and conventions that let independent processes (called *nodes*) on the same machine (or on a network) talk to each other over typed message channels (called *topics*).

The main concepts:

| Concept | Definition |
|---|---|
| **Node** | A process that participates in the ROS2 graph. Usually one node per logical component (camera driver, perception, controller, safety, logger). |
| **Topic** | A named, typed message channel. Many-to-many publish/subscribe. |
| **Message** | A typed data structure (e.g. `geometry_msgs/msg/Twist` for velocity commands, `sensor_msgs/msg/Image` for camera frames). |
| **Publisher** | A node that emits messages on a topic. |
| **Subscriber** | A node that receives messages on a topic. |
| **Service** | RPC-style request/response between two nodes (less common than topics). |
| **Action** | Long-running, cancelable goal with feedback (e.g. "drive to coordinate (5, 2)"). |
| **Parameter** | Per-node configuration value, settable at launch or runtime. |
| **TF2** | Coordinate-frame tree (world → odom → base_link → camera). Tracks transforms over time. |

A typical robot graph looks like:

```
[camera_driver]──> /image_raw ─>[detector]──> /detections ─>[planner]──> /cmd_vel ─>[motor_driver]
                                                                              ^
                                                          [safety_node] ──────┘ (gates /cmd_vel)
```

Every arrow is a topic. Each box is a node, often in its own process / container / language (C++, Python).

---

## 2. ROS1 vs ROS2

- **ROS1** (Kinetic, Melodic, Noetic) — original, 2007-2020s. Single master node; centralized.
- **ROS2** (Foxy, Humble, Iron, Jazzy) — modern; DDS-based; no master node; better for real robots and multi-host setups.

For new projects in 2026: **ROS2 Humble or Iron** are the LTS choices.

---

## 3. Install (when you are ready)

This chapter does NOT require installing ROS2. If you want to:

- **Ubuntu 22.04**: `sudo apt install ros-humble-desktop` (after adding the ROS2 apt repo).
- **Docker**: `docker run -it ros:humble` is the fastest way to try ROS2 without polluting your system.
- **Windows / macOS**: best inside Docker / WSL2 / a VM.

The official tutorials at https://docs.ros.org/en/humble/Tutorials/ are the canonical learning resource.

---

## 4. A publisher / subscriber in ROS2 Python (sketch)

```python
# perception_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception')
        self.pub = self.create_publisher(Float32MultiArray, 'observation', 10)
        self.timer = self.create_timer(0.05, self.tick)

    def tick(self):
        msg = Float32MultiArray()
        msg.data = [distance_to_goal, bearing_to_goal, nearest_obstacle_distance, confidence]
        self.pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(PerceptionNode())
    rclpy.shutdown()
```

```python
# decision_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist

class DecisionNode(Node):
    def __init__(self):
        super().__init__('decision')
        self.sub = self.create_subscription(Float32MultiArray, 'observation', self.cb, 10)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)

    def cb(self, msg):
        d, b, obs, conf = msg.data
        t = Twist()
        t.linear.x = compute_forward(d, b, obs)
        t.angular.z = compute_turn(b)
        self.pub.publish(t)
```

The two run as separate processes; ROS2 wires them together over the `observation` and `cmd_vel` topics. A third node (`safety_node`) can re-publish a gated `cmd_vel_safe` topic that the motor driver subscribes to instead.

---

## 5. Why the toy loop already maps to ROS2

The course's `project_06_physical_ai_simulation` already has the right *shape*:

| Toy code | ROS2 equivalent |
|---|---|
| `GridWorldPerception.observe()` | Perception node publishing `/observation` |
| `Observation` dataclass | `msg.data` payload (or a custom message) |
| `GoalSeekDecision.decide()` | Decision node subscribing `/observation`, publishing `/cmd_vel` |
| `DiffDriveController.control()` | Controller node subscribing `/cmd_vel`, publishing `/wheel_cmd` |
| `SafetyGate.gate()` | Safety node subscribing `/wheel_cmd`, publishing gated `/wheel_cmd_safe` |
| `Robot.step()` | Simulator subscribing `/wheel_cmd_safe`, publishing `/odom` |

The loop is the same; ROS2 turns Python calls into asynchronous topic publishes. The benefit is *language and process independence* — perception can be C++ on a CUDA-using node, decision can be Python, safety can be a certified C++ node.

---

## 6. A mock-ROS pub/sub in plain Python

For the lab, the course provides a pattern that mimics ROS2 pub/sub without ROS2:

```python
class MockNode:
    def __init__(self, bus): self.bus = bus
    def publish(self, topic, msg): self.bus.publish(topic, msg)
    def subscribe(self, topic, cb): self.bus.subscribe(topic, cb)

class Bus:
    def __init__(self): self._subs: dict[str, list] = {}
    def publish(self, topic, msg):
        for cb in self._subs.get(topic, []):
            cb(msg)
    def subscribe(self, topic, cb):
        self._subs.setdefault(topic, []).append(cb)
```

`labs/lab_09_mock_ros_loop.md` walks through wrapping the toy loop in this mock so you experience the publisher/subscriber pattern without installing ROS2.

---

## 7. The simulator side: NVIDIA Isaac, Gazebo, PyBullet

When you eventually move from the toy loop to a real simulator:

- **NVIDIA Isaac Sim** — pairs naturally with ROS2 + Isaac ROS for production-grade robot dev. CUDA-heavy; uses Omniverse.
- **Gazebo / Gazebo Sim** — open-source, ROS2-native; standard in academic robotics.
- **PyBullet** — Python physics for quick prototypes (RL, manipulation).
- **MuJoCo** — research-grade physics, especially for locomotion and dexterous manipulation.

Each ships ROS2 bridges (or works directly with ROS2 messages). Picking one is a hardware + license + integration choice, not a fundamental capability difference at the level of this course.

---

## 8. What you should be able to do after this chapter

- Define node, topic, message, service in your own words.
- Sketch how the toy `project_06` loop maps onto a ROS2 graph of 4-5 nodes.
- Write a tiny pub/sub bus in Python (see `lab_09`).
- Read a simple ROS2 Python node and identify the publisher / subscriber.
- Decide whether your final project needs ROS2 (most do not need it).

---

## 9. Files produced by this chapter

- `docs/17_ros2_robot_sim_intro.md` — this file.
- `labs/lab_09_mock_ros_loop.md` — guided lab: wrap the toy loop in a mock ROS-style bus.
