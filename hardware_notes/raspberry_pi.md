# Hardware notes: Raspberry Pi

This note covers the Pi 4 and Pi 5 as edge AI devices. The Pi is the most popular SBC for prototyping and is widely deployed in light camera / sensor AI use cases, but it has no on-die neural accelerator — for anything beyond classification or simple detection, plan to attach an accelerator (Hailo-8L or Google Coral) or stay on small INT8 models.

---

## When to use a Raspberry Pi

Good fit:
- Light classifier on a USB or CSI webcam (MobileNet, EfficientNet-Lite).
- Sensor pipeline reading I²C / SPI sensors, posting MQTT alerts.
- IoT gateway running ONNX Runtime CPU or TFLite.
- Anything where cost and ecosystem matter more than raw FPS.

Not a good fit:
- Real-time YOLO at 30 FPS without an accelerator.
- Large vision models or LLMs.
- Multi-camera setups beyond 1-2 streams.

If you need real-time CV on a small device, prefer Jetson Orin Nano (see `nvidia_jetson.md`).

---

## Hardware variants worth knowing

| Variant | RAM | CPU | Notes |
|---|---|---|---|
| Raspberry Pi 4B | 1/2/4/8 GB | ARM Cortex-A72 @ 1.5 GHz | Stable, widely deployed |
| Raspberry Pi 5 | 4/8 GB | ARM Cortex-A76 @ 2.4 GHz | ~2-3× faster than Pi 4 for CPU inference, PCIe 2.0 for accelerator HATs |
| Raspberry Pi Zero 2 W | 512 MB | ARM Cortex-A53 | Too small for most AI; usable for TFLite Micro-like workloads |

For this course, **Pi 5 with 8 GB** is the sensible target. Pi 4 still works for sensor and lightweight CV.

---

## Accelerator HATs

A bare Pi 5 produces roughly 2-4 TOPS-equivalent throughput on CPU. With an accelerator HAT attached over PCIe or USB:

| Accelerator | Form factor | Throughput | Runtime |
|---|---|---|---|
| Google Coral USB / M.2 | USB / M.2 | ~4 TOPS INT8 | TFLite Edge TPU |
| Hailo-8L | PCIe M.2 HAT | ~13 TOPS INT8 | Hailo SDK + ONNX Runtime |
| Hailo-8 (full) | PCIe M.2 HAT | ~26 TOPS INT8 | Hailo SDK + ONNX Runtime |

The Pi 5 + Hailo-8L combo is the most popular "budget edge AI" stack as of 2026, and it can sustain real-time YOLO at smaller resolutions, though raw throughput still trails a Jetson Orin Nano.

---

## Software setup

Raspberry Pi OS (64-bit Bookworm) is the recommended base.

```bash
# Update
sudo apt update && sudo apt upgrade -y

# Python and venv
sudo apt install -y python3-pip python3-venv libatlas-base-dev libopenblas-dev

# Create a project venv
python3 -m venv ~/edge-ai-venv
source ~/edge-ai-venv/bin/activate

# Install course deps (lightweight subset — no torchvision-cuda on Pi)
pip install --upgrade pip
pip install numpy pandas matplotlib pillow
pip install onnxruntime              # CPU-only ONNX Runtime
pip install tflite-runtime            # smaller than full tensorflow

# OpenCV
sudo apt install -y python3-opencv

# For camera (CSI)
sudo apt install -y python3-picamera2
```

For Hailo or Coral, follow the vendor's separate install (Hailo SDK, libedgetpu). Their packages are not in PyPI.

---

## Camera options

- **CSI camera (Raspberry Pi Camera Module v2 / v3, HQ Camera)** — much lower latency than USB; use `picamera2` library or `libcamera-vid` → V4L2 → OpenCV.
- **USB webcam** — easy but slower; use OpenCV `cv2.VideoCapture(0)`.
- **IP / RTSP** — works the same as on a laptop.

For Chapter 9 (real-time camera AI), CSI is the realistic deployment target on a Pi; USB is fine for development.

---

## Performance expectations

Approximate end-to-end FPS for a small model (MobileNet-V3 224×224, INT8 TFLite, 640×480 USB webcam) on a stock Pi 5:

| Configuration | FPS (end-to-end) |
|---|---|
| Pi 5 CPU + TFLite INT8 | 15-25 |
| Pi 5 + Coral USB + TFLite Edge TPU INT8 | 30-60 |
| Pi 5 + Hailo-8L + Hailo runtime | 60-120 |

These are rough numbers — your own benchmark using Chapter 4's template is what counts.

---

## Power and thermals

- Pi 5 idle: ~2 W.
- Pi 5 under sustained inference: 8-12 W.
- Pi 5 + Hailo-8L: add ~2-3 W.
- **Active cooling matters.** Without a heatsink + fan, the Pi 5 throttles within minutes. The standard official active cooler is enough for sustained inference.

For battery-powered deployments, plan for at least a 10 W power budget over the full run.

---

## Common pitfalls

- **Building PyTorch from source on Pi.** Use ONNX Runtime or TFLite instead. PyTorch CPU on Pi is slow and the wheel availability is hit-or-miss.
- **Running FP32 models.** Always quantize to INT8 (Chapter 8). Without quantization, Pi performance is disappointing.
- **No swap / out-of-memory.** Add 2 GB of swap on the SD card if you load larger models, but expect SD wear.
- **Forgetting active cooling.** The Pi 5 throttles aggressively above 80 °C.

---

## Where to go next

- Chapter 7 — TFLite runtime on Pi.
- Chapter 8 — quantization (essential here).
- Chapter 9-11 — camera and sensor projects target Pi as a realistic deployment device.
