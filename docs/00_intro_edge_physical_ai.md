# Chapter 0 — Edge AI and Physical AI overview

> **Goal:** Understand what Edge AI is, what Physical AI is, how they differ from Cloud AI / Server AI, and why this direction matters in real-world AI deployment.

This is the opening chapter. It does not yet write code. It establishes the vocabulary, the mental model, and the application landscape that the rest of the course assumes.

---

## 1. Cloud AI vs Edge AI

Most introductory AI courses train and serve models on a powerful cloud or server: a GPU machine, a managed inference endpoint, a Kubernetes cluster. **Cloud AI** means the model lives in a data center, and any device that needs a prediction sends its input over a network and waits for a response.

**Edge AI** flips this: the model runs on or near the device that produces the data — a phone, a Jetson, a Raspberry Pi, an industrial gateway, a camera module, a microcontroller. The model still has to be trained somewhere (usually still the cloud), but at inference time the data does not need to leave the device.

The trade-off is not about model quality — it is about **where the inference happens** and what that implies for latency, bandwidth, privacy, power, and resilience.

### Latency, bandwidth, privacy: typical numbers

Published research and industry surveys in 2024-2025 consistently report a one-to-two-orders-of-magnitude latency gap between edge and cloud paths:

- **Edge inference round-trip:** typically a few milliseconds to ~50 ms, depending on hardware. For example, NVIDIA Jetson-class devices running a quantized YOLO can hit tens of FPS at ~10-30 ms per frame.
- **Cloud inference round-trip:** typically 30-60 ms inside a region, and 100-500 ms once you cross networks, fall back through congested links, or wait on inference cold-starts.

For high-bandwidth signals — 4K video, multi-camera setups, dense sensor arrays — sending raw data to the cloud is expensive in both monetary cost and energy. Edge AI ships only the *decision* (or the small region of interest), not the full stream.

### When the cloud is actually fine

Not every AI system needs to be on the edge. Tasks that:

- aggregate historical data across many devices (fraud detection, recommendation, analytics),
- need very large models that cannot fit on the device,
- can tolerate seconds of latency,
- and where data privacy / network availability are not constraints,

are usually better served from the cloud. The right architecture is often **hybrid**: edge handles the latency-sensitive, privacy-sensitive path, and the cloud handles aggregation, retraining, and large-scale orchestration. This course teaches the edge side of that hybrid; cloud serving is left to specialized cloud-ML courses.

See `figures/cloud_vs_edge.png` for a side-by-side diagram.

---

## 2. On-device inference

**On-device inference** means a trained model is loaded and executed entirely on the target device, without round-trips to a server. The training step still typically happens on a server or cloud GPU; only inference is local.

The constraints that show up at this point are:

- **Memory:** The model file, its activations, and its runtime have to fit in RAM. A 1.5 GB model is not deployable on a 1 GB RPi.
- **Compute:** The device may have only CPU, or a small NPU, or a Jetson-class GPU. The math of the model has to match the hardware.
- **Power and thermals:** On battery-powered or fan-less devices, sustained inference heats up the chip; the OS throttles the clock; latency goes up; the device might shut down.
- **Storage:** Some devices ship with 16-64 MB of flash, not gigabytes.
- **Numerical precision:** Most edge runtimes prefer FP16 or INT8 to FP32 — the model has to tolerate quantization.

The course spends Parts 1-2 (Chapters 1-8) building the toolbox to fit a model into these constraints: benchmarking, exporting to ONNX / TorchScript / TFLite, picking a runtime, and quantization / pruning / distillation.

---

## 3. Real-time AI: latency, throughput, FPS

"Real-time" is a loaded word. In this course we use three concrete metrics:

| Metric | Definition | Why it matters |
|---|---|---|
| **Latency** | Time from input arriving to output ready. Reported as mean and percentiles (P50 / P95 / P99) | An autonomous system that decides at P50 = 20 ms but P99 = 600 ms still has a 600 ms failure mode |
| **Throughput** | Number of inputs the system can process per second under steady load | Limits how many cameras / sensors a single device can support |
| **FPS** | For camera/video pipelines: frames per second processed end-to-end (capture + preprocess + inference + postprocess + display) | Tells you whether the pipeline is dropping frames or keeping up with the camera |

A common beginner mistake is to report only **model-only latency** (the `model(x)` call). Real systems also pay for capture, resize, normalize, postprocess, draw overlay, send alert, and publish a command. The course teaches **end-to-end latency** measurement, defined as:

```
end_to_end_latency =   camera_capture
                     + preprocessing
                     + model_inference
                     + postprocessing
                     + visualization_or_output
                     + communication_or_action_delay
```

Chapter 4 introduces the benchmark template; every project in this repo uses it.

---

## 4. Privacy and offline inference

Edge AI matters wherever data is sensitive or the network cannot be trusted:

- **Camera feeds** (homes, schools, hospitals, factories) — sending raw images out is a privacy and compliance issue.
- **Audio** (smart speakers, hearing aids) — keyword spotting on-device avoids streaming microphone data.
- **Healthcare devices** (wearables, glucose monitors, ECG patches) — patient data should not leave the device.
- **Industrial deployments** (factory floor, oil rig, mining site, ship, plane) — connectivity is intermittent or absent.
- **Autonomous machines** (drone over a remote field, AGV in a warehouse, robot in a disaster zone) — must keep deciding when the network drops.

The course treats privacy and offline operation as first-class design requirements in Chapter 1, and again in Chapter 19 (security / reliability / EdgeOps).

---

## 5. Physical AI: perception → state → decision → action

So far everything has been about *perception-only* AI: a model that classifies, detects, segments, or transcribes. **Physical AI is AI that also acts in the physical world.**

NVIDIA's working definition, which the course adopts: *Physical AI enables autonomous machines to perceive, understand, and perform complex actions in the real (physical) world.* It extends generative / perceptual AI with an understanding of spatial relationships and the physics of the 3D world, taking multimodal inputs (images, video, text, speech, sensors) and converting them into actions an autonomous machine can execute.

The minimum closed loop for Physical AI has six stages:

```
sensors  →  perception  →  state  →  decision  →  controller  →  actuator  →  environment
                                                        ↑                              │
                                                        └──────── feedback ────────────┘
```

- **Sensors:** cameras, LiDAR, IMUs, joint encoders, force sensors, microphones.
- **Perception:** turn raw sensor data into a structured world description (objects, distances, poses, free space).
- **State:** the system's belief about its own situation (pose, velocity, charge level, mode).
- **Decision:** rule-based logic, a learned policy, or — increasingly — a vision-language-action (VLA) model that produces an action.
- **Controller:** translates the decision into low-level commands (joint torques, motor PWM, gripper force).
- **Actuator:** the physical output — wheels, arm joints, gripper, valves, speakers.
- **Feedback:** sensors observe the new state of the environment after the action, and the loop closes.

A **safety layer** sits in front of the actuator: any action command that violates a boundary (joint limit, speed limit, collision risk, low battery, low confidence) is suppressed or replaced with a safe fallback. Safety is non-negotiable for Physical AI; Chapter 15 develops it in detail.

See `figures/physical_ai_loop.png` for the diagram.

**Key distinction:** A camera that classifies images is Edge AI. A robot that takes that classification, decides to move, sends a motor command, observes the result, and adjusts — that is Physical AI. Edge AI is a prerequisite for Physical AI but not the same thing.

---

## 6. Where Edge AI sits among related terms

The phrase "AI on small devices" hides several different scales of system. The course uses these terms precisely:

| Term | Where it runs | Model size / hardware | Typical role |
|---|---|---|---|
| **Cloud AI** | Data-center GPU / TPU | Up to multi-billion-parameter models | Centralized training, large analytics, hosted inference |
| **Edge AI** | On-device or near-device (Jetson, Raspberry Pi, Intel NUC, smart camera, gateway) | Hundreds of MB to a few GB | Real-time, low-latency inference close to the data source |
| **Embedded AI** | Inside a dedicated-function device (sensor, appliance, vehicle ECU) | Task-specific, often custom | Localized intelligence (filtering, simple classification, anomaly flagging) |
| **TinyML** | Microcontroller class (Cortex-M, ESP32, Arduino BLE) | Tens to hundreds of KB, INT8-only | Always-on, battery-powered, ultra-low-power inference |
| **Robotics AI / Autonomous Systems** | A robot or vehicle stack with sensors and actuators | Edge AI + planning + control + safety | Physical AI in production-grade form |
| **Physical AI** | Any AI system whose outputs drive an action in the physical world | Often a stack of Edge AI + controller + safety | Perception-decision-action loops |

These categories overlap. A factory line camera running a defect detector on a Jetson Orin is *Edge AI*. The same Jetson Orin running a controller that sends "reject this product" to a pneumatic arm is part of a *Physical AI* system. The TinyML keyword spotter inside a wearable is both *Embedded AI* and *TinyML*.

In this course, we treat **Edge AI** as the foundation (Parts 1-3) and **Physical AI** as the next step (Part 4), with TinyML and embedded use cases as application chapters.

---

## 7. Where you'll see Edge AI and Physical AI in real applications

The course chapters use this application map. You will revisit it in Chapter 1 (system design) and Chapter 20 (final project).

| Application group | Representative use cases | Typical hardware |
|---|---|---|
| **Camera AI** | Classifier on a webcam, YOLO real-time detection, people counting, helmet detection, ANPR | Laptop CPU/GPU → Jetson → Intel NPU |
| **Sensor AI / Industrial IoT** | Vibration anomaly detection, predictive maintenance, current/voltage monitoring | Raspberry Pi, Intel NUC, microcontroller |
| **TinyML** | Keyword spotting, gesture recognition, simple sensor classification | Arduino Nano 33 BLE, ESP32, Cortex-M MCUs |
| **Smart factory / quality inspection** | Visual defect inspection, robot vision, pick-and-place | Jetson, Intel CPU/NPU, industrial PC |
| **Healthcare devices** | ECG / SpO2 anomaly flagging, fall detection, hearing aids with on-device denoising | Embedded SoC, MCU |
| **Robotics / drone / AGV** | Obstacle avoidance, pose estimation, navigation, manipulation | Jetson, ROS2-based stacks |
| **Autonomous vehicles (small-scale in this course)** | LiDAR / camera perception, lane keeping, sim-to-real | Jetson, NVIDIA Drive (out of course scope), Isaac Sim |
| **Smart cameras / retail** | Customer counting, queue analytics, shelf monitoring | Intel CPU/NPU, Google Coral, Jetson Nano |

---

## 8. What you should be able to do after this chapter

- Explain Edge AI and Physical AI in your own words.
- Distinguish Edge AI from Cloud AI, Embedded AI, TinyML, and Robotics AI.
- Cite at least one real-world use case for each application group above.
- Identify why a given application *cannot* be cloud-only (latency, bandwidth, privacy, offline operation, or physical control).
- Draw the Physical AI closed loop from memory (sensors → perception → state → decision → controller → actuator → feedback) and place a safety layer in it.

---

## 9. Further reading

- NVIDIA Glossary: [What is Physical AI?](https://www.nvidia.com/en-us/glossary/generative-physical-ai/) — the canonical industry definition used in this course.
- IBM: [Edge AI vs Cloud AI](https://www.ibm.com/think/topics/edge-vs-cloud-ai) — accessible overview of the trade-offs.
- Edge Impulse: [Introduction to Edge AI and TinyML](https://www.edgeimpulse.com/) — practical TinyML perspective.
- Survey: *Edge AI vs Cloud AI: A Comparative Study of Performance, Latency and Scalability* (IJRMEET, March 2025) — latency and bandwidth measurements.

---

## 10. Files produced by this chapter

- `docs/00_intro_edge_physical_ai.md` — this file.
- `figures/cloud_vs_edge.png` — side-by-side cloud vs edge inference path diagram.
- `figures/physical_ai_loop.png` — the perception → state → decision → action closed loop with safety layer.
