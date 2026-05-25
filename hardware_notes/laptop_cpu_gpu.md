# Hardware notes: Laptop CPU / GPU

This is the default development device for this course. Every chapter and project runs at Level 1 on a laptop. You can complete almost the entire syllabus without buying any other hardware.

---

## Why use a laptop as an "edge device"

The laptop is not edge hardware in production — but it is the cheapest, fastest way to learn the deployment workflow:

- Same Python tooling as the cloud (PyTorch, ONNX Runtime, OpenCV).
- Both CPU and (often) a discrete or integrated GPU for direct comparison.
- A webcam built-in for Ch 9-10 camera demos.
- No need to provision a Pi or wait for a Jetson.

The mental model: develop and benchmark on the laptop, then **port** to the real edge device by changing the runtime (Ch 6-8) and re-running the benchmark (Ch 4). The latency numbers will be different — but the pipeline will not.

---

## Recommended baseline setup

| Component | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 22.04+, macOS 13+, Windows 10+ with WSL2 | Linux native |
| RAM | 8 GB | 16-32 GB |
| Disk | 30 GB free | 100 GB free |
| CPU | x86_64, 4 cores | 8+ cores |
| GPU | Integrated | NVIDIA discrete GPU with CUDA 11+ |
| Webcam | Any UVC webcam | Built-in or USB |
| Python | 3.10-3.11 | 3.11 |

If you do not have an NVIDIA GPU, you will work with PyTorch CPU and ONNX Runtime CPU — everything still works, just slower. **An integrated Intel iGPU can be targeted with OpenVINO** (Chapter 7) and that path is genuinely useful even on a laptop.

---

## Software install (Linux / WSL2 reference)

```bash
# 1. Conda environment (matches the course env)
conda env create -f environment.yml
conda activate edge-ai

# 2. Verify PyTorch CPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 3. Optional: NVIDIA GPU (CUDA 12.x)
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision

# 4. Verify ONNX Runtime
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"

# 5. Webcam test (OpenCV)
python -c "import cv2; cap = cv2.VideoCapture(0); ok, f = cap.read(); print('webcam OK:', ok, f.shape if ok else None)"
```

If `torch.cuda.is_available()` is `False`, you will be running on CPU. That is fine for most of the course; Chapter 4 benchmarking will simply report higher latencies.

---

## Caveats when treating a laptop as edge

The laptop will *over-state* edge performance because:

1. **Power and thermals are unconstrained.** A laptop on AC has tens of watts of headroom; a Jetson Orin Nano has ~15 W; an MCU has ~50 mW. Latencies you measure on a laptop will be optimistic for any battery-powered or fan-less target.
2. **The CPU/GPU is much beefier.** A laptop discrete GPU has 10-50× more memory than a Jetson Nano. A model that fits comfortably on your laptop GPU may not load on the edge device at all.
3. **Background load is variable.** Browser tabs, IDE, Zoom calls — anything else running will distort benchmarks. Close them, or run benchmarks under `nice -n -10` after closing GUI apps.
4. **USB webcam latency differs from MIPI CSI.** A USB webcam adds 30-100 ms of capture latency that a CSI camera on a Pi or Jetson does not pay.

**Mitigation:** When you report a benchmark, label it `laptop-CPU` or `laptop-GPU`. Do not claim a laptop number as an "edge" number.

---

## When to leave the laptop

Move to a real edge device when:

- You need to measure honest power, thermals, or battery life.
- You need to test integration with a real camera, sensor, robot, or MQTT broker.
- Your final project is targeted at a specific deployment device.
- You want to test sim-to-real on a Jetson (Ch 16-17).

Before then, the laptop is more than enough.

---

## Common pitfalls

- **Mixing CPU and CUDA timings without labeling them.** Always tag every benchmark row with its provider.
- **Forgetting to warm up the GPU before timing.** First inference on CUDA is dominated by kernel compilation; throw away the first ~10-20 iterations. Chapter 4 covers this in the benchmark template.
- **Webcam autoexposure adding 100+ ms.** Lock exposure if you are benchmarking camera latency.
- **Using `time.time()` instead of `time.perf_counter()`.** Always use `perf_counter` for inference timing.

---

## Where to go next

- Chapter 3 — load a PyTorch model and run inference on this laptop.
- Chapter 4 — benchmark properly (warm-up, P95, end-to-end vs model-only).
- Chapter 5-6 — export to ONNX and re-run inference on ONNX Runtime, still on this laptop.
- Chapter 7 — only when you have the matching hardware: TensorRT, OpenVINO, TFLite.
