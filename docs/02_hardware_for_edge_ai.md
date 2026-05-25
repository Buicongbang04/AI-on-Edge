# Chapter 2 — Hardware for Edge AI

> **Goal:** Understand the common classes of edge hardware and pick the right device for a given task. By the end of this chapter you should know when to reach for a microcontroller, a single-board computer (SBC), an edge GPU, or an Intel NPU/iGPU, and what runtime each one expects.

This chapter is a *mapping* chapter — it connects the system-design constraints from Chapter 1 (latency budget, memory, power, cost) to actual physical devices and the runtimes they prefer.

---

## 1. The accelerator landscape

At the level of silicon, edge AI runs on five kinds of compute:

| Accelerator | Strength | Weakness | Found in |
|---|---|---|---|
| **CPU** | Universal; any framework; predictable | Slowest per watt; best for batch=1, small models | Every device |
| **GPU** | Best raw throughput; mature CUDA ecosystem | Hot, power-hungry, expensive | NVIDIA Jetson, gaming laptops, workstations |
| **NPU** (neural processing unit) | Excellent performance per watt; INT8-first | Limited operator coverage; vendor-specific compilers | Intel Core Ultra, Apple M-series, Qualcomm, MediaTek, Hailo |
| **TPU** (tensor processing unit) | Excellent INT8 throughput; very low power | Constrained operator set; small model size limit | Google Coral (Edge TPU) |
| **MCU** (microcontroller) | Microwatts to milliwatts; always-on | Tiny RAM/Flash; INT8-only; weeks of optimization to fit | Cortex-M / RISC-V / ESP32 boards |

A modern edge SoC often contains **several** of these on one die. For example:

- Jetson Orin Nano = ARM CPU + Ampere GPU + 2× DLA (deep learning accelerator).
- Intel Core Ultra = x86 CPU + Iris Xe iGPU + NPU.
- Apple M-series = ARM CPU + integrated GPU + Neural Engine.
- Raspberry Pi 5 = ARM CPU only (no AI accelerator on-die; needs an add-on like Hailo-8L or Coral TPU).

When you "deploy on a Jetson", you have to choose **which** silicon block to target: CPU for compatibility, GPU via CUDA/TensorRT for throughput, or DLA for power efficiency.

---

## 2. Device classes

For practical purposes this course groups edge devices into four classes. Pick the class first; pick the specific device second.

### 2.1 Microcontroller (TinyML class)

- **What it is:** A small chip with KB-to-MB of RAM, MHz-class clock, no OS or a tiny RTOS.
- **Examples:** Arduino Nano 33 BLE Sense, ESP32-S3, Raspberry Pi Pico, STM32 Cortex-M, Sony Spresense.
- **Power:** mW; battery-powered for months.
- **AI runtime:** TensorFlow Lite Micro, EdgeImpulse, X-CUBE-AI.
- **Model size:** 10-500 KB, INT8 only, often quantized + pruned + distilled.
- **Use cases:** Keyword spotting ("Hey Siri"-class), gesture recognition, vibration anomaly, simple sensor classification.
- **Course chapter:** [Chapter 12 — TinyML](12_tinyml_microcontrollers.md).

### 2.2 Single-board computer (SBC class)

- **What it is:** A small Linux board, gigabyte-class RAM, CPU only or CPU+modest GPU/NPU.
- **Examples:** Raspberry Pi 5, Orange Pi 5, BeagleBone, Rock 5.
- **Power:** 3-15 W.
- **AI runtime:** TFLite, ONNX Runtime CPU, Hailo SDK / Edge TPU runtime when an accelerator is attached.
- **Model size:** 1-100 MB models comfortably; large vision models need an accelerator.
- **Use cases:** Light camera AI, sensor AI, IoT gateway, prototyping. Not great for real-time YOLO at 30 FPS *without* an accelerator.
- **Course chapter:** [hardware_notes/raspberry_pi.md](../hardware_notes/raspberry_pi.md).

### 2.3 Edge GPU class

- **What it is:** A small board with a real CUDA GPU on-die.
- **Examples:** NVIDIA Jetson Nano / Orin Nano / Orin NX / Orin AGX.
- **Power:** 5-60 W (configurable nvpmodel).
- **AI runtime:** PyTorch CUDA, ONNX Runtime GPU, TensorRT FP16/INT8.
- **Model size:** Hundreds of MB to a few GB models possible; YOLOv8n runs real-time on Orin Nano.
- **Use cases:** Real-time camera AI, multi-stream, robotics, ROS2.
- **Course chapter:** [hardware_notes/nvidia_jetson.md](../hardware_notes/nvidia_jetson.md).

### 2.4 Industrial PC / x86 edge class

- **What it is:** A small x86 box with CPU, iGPU, and increasingly an NPU.
- **Examples:** Intel NUC 11/13/14, Lattepanda Sigma, mini-PCs, ruggedized industrial PCs.
- **Power:** 15-65 W.
- **AI runtime:** OpenVINO (preferred on Intel hardware), ONNX Runtime CPU/CUDA, plus optional Hailo accelerator.
- **Model size:** Comparable to Jetson; better for large LLMs offloaded to NPU.
- **Use cases:** Smart factory, retail kiosk, edge LLM, multi-model heterogeneous workloads.
- **Course chapter:** [hardware_notes/intel_openvino.md](../hardware_notes/intel_openvino.md).

### 2.5 Bonus: Laptop CPU/GPU as your default development device

For most learners in this course, the *first* deployment target is **your laptop**. It is not "true" edge hardware, but it is the cheapest way to learn the deployment workflow before buying anything. The whole repo runs at Level 1 on a laptop. See [hardware_notes/laptop_cpu_gpu.md](../hardware_notes/laptop_cpu_gpu.md).

---

## 3. Trade-off triangle: cost / power / compute

You almost never get all three. Pick two:

```
            high compute
                  ^
                  │
          Jetson Orin AGX    Intel NUC + Hailo
                  │
                  │
                  │    Jetson Orin Nano
                  │
                  │
                  │
      <───────────┼───────────>  low cost
                  │
                  │
      Raspberry Pi 5
                  │
        Coral TPU stick
                  │
        ESP32 / Cortex-M
                  v
            low power
```

(Conceptual sketch; specific positions vary by workload.)

A learner who wants **low cost + low power** lands on a Pi 5 or a microcontroller, and pays in compute. A learner who wants **high compute + low power** ends up at Jetson Orin Nano or an NPU-equipped Intel system, and pays in dollars. A learner who wants **high compute + low cost** ends up on a desktop / laptop GPU, and pays in power.

---

## 4. Quick reference: which device for which problem

Adapted from the Instruction matrix and validated against 2026 benchmark surveys.

| Use case | Recommended device | Recommended runtime | Notes |
|---|---|---|---|
| Real-time YOLO at 30 FPS, single camera, 720p | Jetson Orin Nano (or Orin NX) | TensorRT FP16 | Outperforms Pi 5 + Coral by ~2× at this workload |
| Real-time YOLO at 30 FPS, multi-camera | Jetson Orin NX / Orin AGX, or Intel NUC + Hailo | TensorRT or OpenVINO | NUC11+Hailo case hit 112 FPS with 30 streams |
| Lightweight classifier, single camera, prototype | Laptop CPU/GPU → Raspberry Pi 5 | ONNX Runtime CPU / TFLite | Often runs fine without accelerator |
| Sensor anomaly detection, low rate | Raspberry Pi 4/5, Intel NUC | ONNX Runtime CPU / OpenVINO | Latency is not the limit; sample rate is |
| Always-on keyword spotting, battery | Arduino Nano 33 BLE / ESP32 / Cortex-M | TensorFlow Lite Micro | Power, not latency, is the constraint |
| Industrial defect inspection, factory line | Intel NUC (Iris Xe + NPU), Jetson Orin Nano | OpenVINO or TensorRT | Choose by existing factory IT stack |
| Edge LLM (7B parameter class) | Intel Core Ultra NUC + OpenVINO NPU offload | OpenVINO 2026 | ~3.8× throughput vs GPU-only on the same box |
| Robot perception + ROS2 | Jetson Orin Nano / Orin NX | TensorRT + ROS2 Humble/Iron | NVIDIA Isaac integration is mature here |
| First-time prototype, no budget | Laptop CPU/GPU | PyTorch + ONNX Runtime CPU | Start here; buy hardware later |

---

## 5. Cameras and sensors

The model is half of the system; the sensor is the other half. Common pairings:

**Cameras:**
- USB webcam (UVC) — simplest; use with OpenCV `VideoCapture(0)`. Latency 30-100 ms because of USB and MJPEG decode.
- MIPI CSI camera (Raspberry Pi Camera v2/v3, IMX219, IMX477) — much lower latency on RPi and Jetson; needs `libcamera` or `nvarguscamerasrc`.
- IP / RTSP camera — for fixed installations; OpenCV `VideoCapture(rtsp://...)`.
- Industrial GigE Vision (Basler, FLIR, Sony) — for factory-grade reliability; needs vendor SDK.
- Stereo cameras (Intel RealSense, ZED, OAK) — for depth, often with on-device perception.

**Other sensors used in this course:**
- IMU (Bosch BMI270, ICM-20948) — for gesture, fall detection, robot pose.
- Vibration (ADXL345, MEMS) — for industrial anomaly detection (Chapter 11).
- Microphone array (ReSpeaker, MEMS) — for keyword spotting (Chapter 12).
- Temperature / humidity / current / pressure — generic IoT sensors over I²C / SPI / MQTT.

**Design rules for sensor selection:**
1. Sensor sample rate must be *at least 2×* the frequency of the signal you care about.
2. Sensor latency adds to inference latency. A 30 FPS USB webcam adds ≥33 ms before you even run the model.
3. Lighting (for cameras) is more important than the camera itself for most CV tasks.
4. If the sensor can preprocess (auto-exposure, denoising), use it — it offloads work from your CPU.

---

## 6. How to map this course's hardware tiers

For each project in this repo, the README lists the hardware tier it targets:

- **Level 1 (laptop CPU/GPU):** must work, mandatory baseline.
- **Level 2 (Raspberry Pi / Intel NUC):** works with TFLite / OpenVINO; usually the realistic edge deployment.
- **Level 3 (Jetson):** real-time camera AI, TensorRT path, multi-stream.
- **Level 4 (microcontroller):** TFLite Micro; usually only Chapter 12.

You do not need to own hardware at every level. The curriculum is structured so that the laptop tier alone takes you through Chapters 0-16. Hardware-specific chapters say so up front.

---

## 7. What you should be able to do after this chapter

- Distinguish microcontroller, SBC, edge GPU, and x86 industrial PC.
- For a given system spec (from Chapter 1), pick a device class and justify it.
- Know which runtime fits which device: PyTorch / ONNX Runtime / TensorRT / OpenVINO / TFLite / TFLite Micro.
- Read the per-hardware notes for the device you actually plan to use.

---

## 8. Files produced by this chapter

- `docs/02_hardware_for_edge_ai.md` — this file.
- `hardware_notes/laptop_cpu_gpu.md` — how to use a laptop as your edge dev device.
- `hardware_notes/raspberry_pi.md` — RPi 4/5 setup, TFLite, optional Hailo/Coral.
- `hardware_notes/nvidia_jetson.md` — Jetson setup, JetPack, TensorRT.
- `hardware_notes/intel_openvino.md` — Intel CPU/iGPU/NPU + OpenVINO.

---

## 9. Sources

- NVIDIA Jetson Orin Nano benchmarks (MLPerf Edge v3.0; JetPack 6.2 reference).
- Intel OpenVINO 2026 release notes — heterogeneous CPU/GPU/NPU partitioning.
- IEEE 2025 conference paper: *Benchmarking Edge AI Platforms — NVIDIA Jetson vs Raspberry Pi 5 with Coral TPU*.
- arXiv 2024: *Benchmarking Edge AI Platforms for High-Performance ML Inference*.
- Hailo, Google Coral, and Edge Impulse product documentation.
