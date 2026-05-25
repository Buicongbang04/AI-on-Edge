# Course overview

**Edge AI and Introductory Physical AI** — From On-device Inference to Perception-Action Loops.

This document describes the curriculum at the *part* level. For the chapter-by-chapter list, see [`SYLLABUS.md`](SYLLABUS.md). For the weekly schedule, see [`ROADMAP.md`](ROADMAP.md).

---

## Position in the AI learning path

```
Python + Math Foundation
        ↓
Machine Learning Foundation
        ↓
Deep Learning Foundation
        ↓
Computer Vision / NLP / Generative AI
        ↓
Edge AI and Introductory Physical AI     ← this course
        ↓
Robotics AI / Embedded AI / Industrial AI / Autonomous Systems
```

This course assumes the learner already trains models. It teaches what comes after training: getting models to run on real devices, reliably, within strict latency/memory/power budgets, and (in Part 4) closing the loop from sensing to action.

---

## The central question

> **How do you take an AI model from a notebook / server / GPU training environment, and make it run stably on a real device, in real time, within the limits of latency, memory, power, temperature, cost, privacy, and reliability?**

The course does not teach modeling. It teaches the full deployment pipeline:

1. Collect data from camera / microphone / sensor / video stream.
2. Train or fine-tune a model that fits the problem.
3. Export the model to ONNX / TorchScript / TFLite or a runtime-specific format.
4. Optimize via quantization, pruning, distillation, FP16 / INT8, and runtime acceleration.
5. Run inference with ONNX Runtime, TensorRT, OpenVINO, TFLite, or TensorFlow Lite Micro.
6. Measure latency, FPS, throughput, memory, power, temperature, and end-to-end response time.
7. Connect the model to a camera, sensor, dashboard, API, MQTT, or actuator / simulator.
8. Design a perception → state → decision → control → action → feedback loop with a safety gate.

---

## The five parts

| Part | Title | Role | Chapters |
|---|---|---|---|
| **Part 1** | Edge AI Foundation | Build the systems mindset; understand device, inference, and benchmark constraints | Ch 0-4 |
| **Part 2** | Model Deployment | Export, runtime, and optimization (PyTorch → ONNX → ONNX Runtime / TensorRT / OpenVINO / TFLite) | Ch 5-8 |
| **Part 3** | Edge Applications | Camera AI, object detection, sensor AI, TinyML, and industrial inspection use cases | Ch 9-13 |
| **Part 4** | Physical AI | Perception → state → decision → controller → action → feedback, plus simulation and ROS2 intro | Ch 14-17 |
| **Part 5** | Advanced Topics and Final Project | Edge LLM / VLM, reliability, monitoring, safety, and the capstone | Ch 18-20 |

---

## What changes from earlier ML / DL courses

| Earlier courses focus on | This course focuses on |
|---|---|
| Loss, accuracy, F1, mAP | Latency P50/P95/P99, FPS, memory peak, power, temperature, utilization |
| `loss.backward()` | `torch.no_grad()`, `model.eval()`, warm-up iterations |
| Train / val / test splits | Train env vs runtime env; export reproducibility |
| Bigger models | Smaller-and-faster models; quantization; distillation |
| `accuracy_score(...)` | End-to-end response time, dropped frames, fallback triggered rate |
| One GPU | CPU, GPU, NPU, TPU, microcontroller — and the trade-off between them |
| Notebook output | A device, a camera feed, an MQTT topic, an actuator, a closed loop, a logged event |

---

## What the learner builds

Across the course, the learner produces:

- **6 mini-projects**: camera classifier, YOLO real-time detector, sensor anomaly detection, industrial quality inspection, TinyML keyword spotting (concept demo), Physical AI simulation loop.
- **One final project** chosen from a list of templates (or a custom problem), with a complete end-to-end pipeline, benchmark report, deployment notes, and risk/safety notes.
- **Benchmark scripts** for latency, memory, and FPS that follow a standard schema.
- **Deployment notes** for each major project: hardware.md, runtime.md, safety.md.

---

## Hardware support tiers

Not every learner has a Jetson or a microcontroller. The repo supports a fallback tier:

| Tier | Hardware | What works |
|---|---|---|
| Level 1 (minimum) | Laptop CPU or GPU | Almost everything: PyTorch / ONNX Runtime inference, camera demos via OpenCV, quantization, sensor pipelines, Physical AI simulation |
| Level 2 | Raspberry Pi / Intel NUC | Lightweight TFLite / ONNX Runtime / OpenVINO inference, real edge benchmarks |
| Level 3 | NVIDIA Jetson | TensorRT, real-time YOLO, FP16/INT8 acceleration, on-device camera AI |
| Level 4 | Microcontroller (Arduino BLE, ESP32, etc.) | TinyML via TensorFlow Lite Micro |
| Optional | Google Coral TPU | TFLite Edge TPU acceleration |

The repo always runs at Level 1. Higher-tier sections are marked, and their commands are isolated in chapter-level `hardware_notes/` files.

---

## Why this matters

For many years, AI was trained and deployed on powerful cloud or server hardware. But many real applications cannot depend entirely on the cloud:

- **Latency**: cameras / sensors generate continuous data; sending all of it to the cloud wastes bandwidth and adds delay.
- **Autonomy under network loss**: robots, drones, autonomous machines, and industrial equipment must decide locally, near real time.
- **Privacy**: image, audio, healthcare, and manufacturing data may be sensitive and must be processed on-device.
- **Cost**: edge devices have limited RAM, CPU/GPU/NPU, power, thermals, and dollar cost.
- **Physical control**: Physical AI does not just perceive; it decides and acts in the physical environment, and that means safety constraints and feedback loops.

Edge AI is the prerequisite for Physical AI, robotics, embedded AI, and autonomous systems. This course is the bridge.

---

## Reading order

If you are new to deployment, follow the chapters in order. If you already know parts of the stack, the following short paths work:

- **"I want to deploy YOLO on Jetson":** Ch 1, Ch 2, Ch 4, Ch 5, Ch 6, Ch 7, Ch 10.
- **"I want to do sensor / TinyML":** Ch 1, Ch 4, Ch 8, Ch 11, Ch 12.
- **"I want to understand Physical AI":** Ch 0, Ch 1, Ch 9, Ch 14, Ch 15, Ch 16, Ch 17.
- **"I want to write a final project right now":** Ch 0, Ch 1, Ch 4, Ch 19, Ch 20.

The final project sits on top of the whole course — it requires having read enough of Parts 1-4 to make defensible design choices.
