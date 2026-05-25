# Hardware notes: NVIDIA Jetson

Jetson modules are the most capable single-board option for real-time computer vision and robotics. They pair an ARM CPU with a CUDA GPU and (on Orin) dedicated DLA accelerators, all of which TensorRT can target. This is the device class the course recommends for any real-time camera AI or Physical AI project.

---

## Jetson generations to know

| Module | RAM | GPU | TOPS (INT8) | Power profiles | Common use |
|---|---|---|---|---|---|
| Jetson Nano (original, 2019) | 4 GB | 128-core Maxwell | ~0.5 | 5/10 W | Entry-level CV (now aging) |
| Jetson Xavier NX | 8/16 GB | 384-core Volta + 48 TC | ~21 | 10/15/20 W | Mid-range CV |
| Jetson Orin Nano (Super, 2024+) | 4/8 GB | 1024-core Ampere + 32 TC | ~40 (Super: ~67) | 7/15/25 W | **Course default** |
| Jetson Orin NX | 8/16 GB | 1024-core Ampere + 32 TC | ~70-100 | 10/15/25 W | Pro CV, multi-stream |
| Jetson Orin AGX | 32/64 GB | 2048-core Ampere + 64 TC | ~200-275 | 15-60 W | Heavy multi-model, robotics |

**For this course, Jetson Orin Nano 8 GB Developer Kit is the sweet spot.** It runs TensorRT, has enough RAM for YOLOv8 + post-processing, and the developer kit comes with a carrier board ready to use.

---

## Software stack: JetPack

JetPack is NVIDIA's SDK bundle for Jetson. It includes the L4T (Linux for Tegra) OS, CUDA, cuDNN, TensorRT, OpenCV, multimedia APIs, and the deepstream SDK.

| JetPack | Ubuntu | CUDA | TensorRT | Recommended for |
|---|---|---|---|---|
| JetPack 5.1.x | 20.04 | 11.4 | 8.5 | Xavier NX, older Orin Nano |
| JetPack 6.0 / 6.1 / 6.2 | 22.04 | 12.x | 8.6 / 10.x | Orin Nano Super, Orin NX, Orin AGX (course default) |

**Always match JetPack to the module.** Installing JetPack 6 on an old Xavier or Nano will not work.

---

## Installing JetPack on Orin Nano Developer Kit (summary)

1. Download the official Orin Nano Developer Kit SD card image from the NVIDIA Developer site (the image name includes the JetPack version).
2. Flash it to a microSD (or to the NVMe SSD via the SDK Manager on a host Ubuntu PC).
3. Boot, complete the Ubuntu first-run wizard.
4. Run `sudo apt update && sudo apt upgrade -y`.
5. Verify: `dpkg -l | grep -i tensorrt` should list TensorRT 10.x; `nvcc --version` should show CUDA 12.x.

Full instructions are in the official [NVIDIA Jetson Developer Guide](https://docs.nvidia.com/jetson/).

---

## Python environment on Jetson

Jetson uses ARM64. Do **not** install PyTorch from generic `pip install torch` — most stock wheels are x86_64 CPU-only. Use NVIDIA's Jetson-specific PyTorch and TensorRT wheels:

```bash
# Pick the wheel that matches your JetPack version
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/
pip install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+...whl
pip install torchvision   # ARM-compatible wheel from same NVIDIA index

# ONNX Runtime GPU for Jetson
pip install onnxruntime-gpu   # Jetson-specific build from NVIDIA index

# TensorRT comes pre-installed by JetPack; use it via `import tensorrt as trt`
```

Confirm:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python3 -c "import tensorrt as trt; print(trt.__version__)"
python3 -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

`get_available_providers()` should include `CUDAExecutionProvider` and ideally `TensorrtExecutionProvider`.

---

## The TensorRT path

The whole point of Jetson is TensorRT. The course path is:

```
PyTorch model (.pt)
    ↓  (torch.onnx.export, Chapter 5)
ONNX model (.onnx)
    ↓  (trtexec or polygraphy, Chapter 7)
TensorRT engine (.engine / .plan)
    ↓
Run via Python (tensorrt + pycuda) or ONNX Runtime TensorrtExecutionProvider
```

`trtexec` (ships with TensorRT) is the simplest way to build an engine from ONNX:

```bash
# FP16 engine for an exported YOLO ONNX
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp16.engine --fp16
```

Add `--int8 --calib=<calib.cache>` for INT8 with calibration (Chapter 8).

---

## Camera options on Jetson

- **MIPI CSI** (IMX219, IMX477, IMX477 HQ) — much lower latency and higher quality than USB. Use `nvarguscamerasrc` GStreamer pipeline or `jetson-utils`.
- **USB webcam** — works the same as on a laptop via OpenCV. Higher latency.
- **Industrial cameras** over GigE or USB3 — supported with vendor SDKs (Basler, FLIR).
- **RealSense / OAK / ZED depth cameras** — well-supported on Jetson and used in ROS2 stacks.

For Chapter 9-10 demos on Jetson, prefer a CSI camera if you have one.

---

## Power profiles

Jetson modules have power profiles (`nvpmodel`) you can switch at runtime:

```bash
# List profiles
sudo nvpmodel -q --verbose
# Example: set 25W max performance mode on Orin Nano Super
sudo nvpmodel -m 0    # 0 = MAXN_SUPER on Orin Nano Super
sudo jetson_clocks    # lock clocks to max
```

When benchmarking, **document which power mode is active** — numbers are not comparable across modes.

---

## Performance expectations

Approximate end-to-end FPS on Orin Nano Super (25 W mode) with TensorRT FP16:

| Workload | FPS (end-to-end) |
|---|---|
| YOLOv8n 640×640 | 60-90 |
| YOLOv8s 640×640 | 30-50 |
| YOLOv8m 640×640 | 10-20 |
| MobileNetV3 224×224 classifier | 200+ |
| Stable Diffusion 1.5 (offline) | 0.5-1 image/s |

Your own benchmark numbers will vary based on power mode, input resolution, and pre/post-processing — use Chapter 4's template.

---

## Thermals and cooling

The Orin Nano Developer Kit's stock fan handles 15 W mode adequately. For sustained 25 W operation:

- Keep the room cool (< 30 °C ambient).
- Consider a third-party heatsink/fan upgrade.
- Watch `tegrastats` for thermal throttling.

If the device passes ~85 °C, the GPU clocks will throttle and your inference latency will spike. This is *the* classic "demo works in the lab, fails in production" trap on Jetson.

---

## Common pitfalls

- **Installing generic PyTorch wheels.** Use the NVIDIA-built ones for ARM + CUDA.
- **Forgetting to warm up before benchmarking.** First TensorRT inference includes engine deserialization; throw away the first ~10 iterations.
- **Mixing JetPack versions.** Once flashed, do not "partially upgrade" CUDA / TensorRT independently.
- **Not running `sudo jetson_clocks`.** Without it, the GPU clocks vary, and benchmark numbers are noisy.
- **Underestimating thermal throttling.** Long runs at 25 W need real cooling.
- **Building large Docker images on the SD card.** Use an NVMe SSD if available; SD cards wear quickly under heavy workloads.

---

## Where to go next

- Chapter 5 — PyTorch → ONNX export (necessary before TensorRT).
- Chapter 6 — ONNX Runtime, including CUDA / TensorRT execution providers.
- Chapter 7 — TensorRT lab (Jetson-only hands-on).
- Chapter 9-10 — real-time camera AI projects target Jetson as the realistic deployment.
- Chapter 17 — ROS2 on Jetson is the bridge to robotics engineering.
