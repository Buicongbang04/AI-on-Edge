# Course overview

**Edge AI and Introductory Physical AI** — *From On-device Inference to Perception-Action Loops*

This document describes the curriculum at the *part* level — what each part teaches, why it exists, and what learners build. For the chapter-by-chapter list with exact file references, see [`SYLLABUS.md`](SYLLABUS.md). For the week-by-week schedule, see [`ROADMAP.md`](ROADMAP.md).

---

## Position in the AI learning path

```
Python + Math Foundation
        |
        v
Machine Learning Foundation
        |
        v
Deep Learning Foundation
        |
        v
Computer Vision / NLP / Generative AI
        |
        v
Edge AI and Introductory Physical AI     <-- this course
        |
        v
Robotics AI / Embedded AI / Industrial AI / Autonomous Systems
```

This course assumes the learner already trains models. It teaches what comes after training: getting models to run on real devices, reliably, within strict latency / memory / power budgets, and (in Part 4) closing the loop from sensing to action.

---

## The central question

How do you take an AI model from a notebook / server / GPU training environment, and make it run stably on a real device, in real time, within the limits of latency, memory, power, temperature, cost, privacy, and reliability — and, when the model is meant to act in the physical world, with the safety boundary and feedback loop that requires?

The course does not teach modeling. It teaches the full deployment pipeline:

1. Collect data from camera / microphone / sensor / video stream.
2. Train or fine-tune a model that fits the problem.
3. Export the model to ONNX / TorchScript / TFLite or a runtime-specific format.
4. Optimize via quantization, pruning, distillation, FP16 / INT8, and runtime acceleration.
5. Run inference with ONNX Runtime, TensorRT, OpenVINO, TFLite, or TensorFlow Lite Micro.
6. Measure latency, FPS, throughput, memory, power, temperature, and end-to-end response time.
7. Connect the model to a camera, sensor, dashboard, API, MQTT, or actuator / simulator.
8. Design a perception -> state -> decision -> control -> action -> feedback loop with a safety gate.
9. Run it in production-style with logging, model versioning, fallback, monitoring, and rollback.

---

## The five parts

| Part | Title | Role | Chapters |
|---|---|---|---|
| **Part 1** | Edge AI Foundation | Build the systems mindset; understand device, inference, and benchmark constraints | Ch 0-4 |
| **Part 2** | Model Deployment | Export, runtime, and optimization (PyTorch -> ONNX -> ONNX Runtime / TensorRT / OpenVINO / TFLite) | Ch 5-8 |
| **Part 3** | Edge Applications | Camera AI, object detection, sensor AI, TinyML, and industrial inspection use cases | Ch 9-13 |
| **Part 4** | Physical AI | Perception -> state -> decision -> controller -> action -> feedback, plus simulation and ROS2 intro | Ch 14-17 |
| **Part 5** | Advanced Topics and Final Project | Edge LLM / VLM, reliability, monitoring, safety, and the capstone | Ch 18-20 |

### What is delivered in each part (what actually ships in this repo)

**Part 1 (Ch 0-4) — Edge AI Foundation.** Concept overview with diagrams (`figures/cloud_vs_edge.png`, `figures/physical_ai_loop.png`); a system-design template (Ch 1 assignment); hardware decision guide with four per-hardware notes (`hardware_notes/`); a clean PyTorch inference reference (`src/inference/infer_pytorch.py`); and a benchmark suite (`src/benchmark/`) with latency, memory, FPS, and a CLI that saves JSON to `experiments/benchmark_results/`.

**Part 2 (Ch 5-8) — Model Deployment.** ONNX export with validation (`src/export/export_onnx.py`); ONNX Runtime inference (`src/inference/infer_onnxruntime.py`) with execution-provider docs; TensorRT and OpenVINO inference templates; an INT8 quantization toolkit (`src/optimization/quantization.py`) with dynamic and static modes and a `compare_models` helper; one lab (`labs/lab_04_tensorrt_or_openvino.md`) and one assignment (`assignments/assignment_04_quantization.md`).

**Part 3 (Ch 9-13) — Edge Applications.** A real-time camera classifier (`src/inference/camera_loop.py`) with FPS counter, frame skipping, recording, and headless mode; a YOLO real-time detector project (`projects/project_02_yolo_realtime_camera/`); a sensor anomaly detection project (`projects/project_03_sensor_anomaly_detection/`) with a runnable train + infer pipeline; a TinyML keyword-spotting project (`projects/project_05_tinyml_keyword_spotting/`) with a footprint estimator; and an industrial quality inspection project skeleton (`projects/project_04_quality_inspection_ai/`).

**Part 4 (Ch 14-17) — Physical AI.** Four small modules implementing the loop (`src/physical_ai/{perception, decision, controller, safety}.py`); a working toy 2D simulation (`projects/project_06_physical_ai_simulation/`) where a differential-drive robot navigates around obstacles to a goal with a safety gate; two labs (`lab_08_robot_simulation_loop.md`, `lab_09_mock_ros_loop.md`); a ROS2 orientation chapter.

**Part 5 (Ch 18-20) — Advanced Topics + Final Project.** An on-device LLM intro chapter with a memory-budget calculator (`notebooks/chapter_11_edge_llm_intro.ipynb`); an EdgeOps toolkit (`src/edgeops/{logging, versioning, fallback}.py`) with a device log schema, SHA-256 model checksum, and a fallback decision helper; deployment-notes templates (`deployment_notes/runtime.md`, `deployment_notes/safety.md`); a final-project template and the 100-point grading rubric (`reports/final_project_report_template.md`).

---

## What changes from earlier ML / DL courses

| Earlier courses focus on | This course focuses on |
|---|---|
| Loss, accuracy, F1, mAP | Latency P50 / P95 / P99, FPS, memory peak, power, temperature, utilization |
| `loss.backward()` | `torch.no_grad()`, `model.eval()`, warm-up iterations |
| Train / val / test splits | Train environment vs runtime environment; export reproducibility |
| Bigger models | Smaller-and-faster models; quantization; distillation |
| `accuracy_score(...)` | End-to-end response time, dropped frames, fallback trigger rate |
| One GPU | CPU, GPU, NPU, TPU, microcontroller — and the trade-off between them |
| Notebook output | A device, a camera feed, an MQTT topic, an actuator, a closed loop, a logged event |

---

## What the learner builds

Across the course, the learner produces:

- **Six mini-projects** in `projects/`:
    1. Camera image classifier (starter, builds on `src/inference/camera_loop.py`).
    2. YOLO real-time camera detector (image + video + ONNX export).
    3. Sensor anomaly detection (synthetic vibration -> train autoencoder -> infer stream -> alerts).
    4. Industrial quality inspection (concept skeleton with dataset suggestions).
    5. TinyML keyword spotting (synthetic 3-class -> train -> footprint estimate).
    6. Physical AI simulation (toy 2D world, perception -> decision -> controller -> safety -> actuator loop).
- **One final project** chosen from a list of templates (or a custom problem) with a benchmark report, deployment notes, and risk / safety notes, graded by the 100-point rubric in `reports/final_project_report_template.md`.
- **Benchmark JSONs** in `experiments/benchmark_results/` produced by `src.benchmark`; one per (model, runtime, hardware) row.
- **Per-project deployment notes** that follow the `deployment_notes/runtime.md` and `deployment_notes/safety.md` templates.

---

## Hardware support tiers

Not every learner has every device. The repo supports a fallback ladder:

| Tier | Hardware | What works |
|---|---|---|
| Level 1 (minimum) | Laptop CPU or GPU | Almost everything: PyTorch / ONNX Runtime inference, camera demos via OpenCV, quantization, sensor pipelines, Physical AI simulation, Edge LLM concept |
| Level 2 | Raspberry Pi / Intel NUC | Lightweight TFLite / ONNX Runtime / OpenVINO inference, real edge benchmarks |
| Level 3 | NVIDIA Jetson Orin Nano (or NX) | TensorRT, real-time YOLO, FP16 / INT8 acceleration, on-device camera AI |
| Level 4 | Microcontroller (Arduino BLE, ESP32, Cortex-M) | TinyML via TensorFlow Lite Micro |
| Optional | Google Coral TPU | TFLite Edge TPU acceleration |

The repo always runs at Level 1. Higher-tier sections are clearly marked, and their commands are isolated in `hardware_notes/` and the per-chapter labs.

---

## Why this matters

For many years, AI was trained and deployed on powerful cloud or server hardware. But many real applications cannot depend entirely on the cloud:

- **Latency:** cameras and sensors generate continuous data; sending all of it to the cloud wastes bandwidth and adds delay.
- **Autonomy under network loss:** robots, drones, autonomous machines, and industrial equipment must decide locally, near real time.
- **Privacy:** image, audio, healthcare, and manufacturing data may be sensitive and must be processed on-device.
- **Cost:** edge devices have limited RAM, CPU / GPU / NPU, power, thermals, and dollar cost.
- **Physical control:** Physical AI does not just perceive — it decides and acts in the physical environment, and that means safety constraints and feedback loops.

Edge AI is the prerequisite for Physical AI, robotics, embedded AI, and autonomous systems. This course is the bridge.

---

## Reading order

If you are new to deployment, follow the chapters in order. If you already know parts of the stack, the following short paths work:

- **"I want to deploy YOLO on Jetson":** Ch 1, 2, 4, 5, 6, 7, 10.
- **"I want to do sensor / TinyML":** Ch 1, 4, 8, 11, 12.
- **"I want to understand Physical AI":** Ch 0, 1, 9, 14, 15, 16, 17.
- **"I want to write a final project right now":** Ch 0, 1, 4, 19, 20.

The final project sits on top of the whole course — it requires having read enough of Parts 1-4 to make defensible design choices.
