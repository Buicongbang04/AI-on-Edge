# Lab 04 — TensorRT or OpenVINO (hardware-dependent)

**Chapter:** 7 — Optimized runtimes
**Prerequisites:** finished Chapters 4-6 (benchmarking, ONNX export, ONNX Runtime).
**Estimated effort:** 2-3 hours per track.

This lab has **two tracks** — pick the one that matches your hardware. If you have both Jetson and an Intel system, do both. If you have neither, do **Track B on Intel CPU** (it works without iGPU/NPU).

---

## Track A — TensorRT on NVIDIA Jetson

### A.1 Prerequisites

- An **NVIDIA Jetson** (Orin Nano / Orin NX / Orin AGX recommended) with **JetPack 6.x**.
- TensorRT 10.x (pre-installed by JetPack).
- The Chapter 5 ONNX file copied to the Jetson.

### A.2 Build a TensorRT engine

1. Copy `experiments/exported_models/mobilenet_v3_small.onnx` to the Jetson.
2. Set the device to its max performance profile so build numbers are stable:

   ```bash
   sudo nvpmodel -m 0       # MAXN_SUPER on Orin Nano Super
   sudo jetson_clocks       # lock clocks
   ```

3. Build three engines: FP32 baseline, FP16, and INT8 (skip INT8 for now if you don't have calibration data — Chapter 8 covers it).

   ```bash
   trtexec --onnx=mobilenet_v3_small.onnx --saveEngine=mobilenet.engine
   trtexec --onnx=mobilenet_v3_small.onnx --saveEngine=mobilenet_fp16.engine --fp16
   ```

   Time how long each build takes. Note how `trtexec` prints layer-by-layer timings during the build search.

### A.3 Run inference

```bash
python src/inference/infer_tensorrt.py --engine mobilenet_fp16.engine --image datasets/sample.jpg
```

Confirm the top-5 predictions match what PyTorch / ONNX Runtime produced. (They may not match exactly at FP16 — but the top-1 class should be the same.)

### A.4 Benchmark and compare

Fill in this table on your Jetson:

| Runtime | Precision | Engine size (MB) | P50 ms | P95 ms | FPS estimate |
|---|---|---|---|---|---|
| PyTorch (CUDA) | FP32 | — | | | |
| ONNX Runtime (CUDAExecutionProvider) | FP32 | — | | | |
| TensorRT | FP32 | | | | |
| TensorRT | FP16 | | | | |

Use Chapter 4's benchmark template for the first two rows, and either `trtexec --loadEngine=...` or the per-script timing for the TensorRT rows.

### A.5 Reflection

In your lab report (max 1 page) answer:

1. How much speedup did TensorRT FP16 give over PyTorch CUDA?
2. How much *smaller* is the FP16 engine vs the FP32 one?
3. Did any prediction change? Was the top-1 stable across all 4 rows?
4. How long did the TensorRT build take? Is that acceptable for your production pipeline (or do you need to cache engines)?

---

## Track B — OpenVINO on Intel hardware

### B.1 Prerequisites

- Any Intel CPU (works at minimum), ideally with Iris Xe iGPU or Core Ultra NPU.
- `pip install openvino openvino-dev`
- (Linux iGPU) `sudo apt install intel-opencl-icd`
- (Core Ultra NPU) install the Intel NPU driver.

### B.2 Check available devices

```python
import openvino as ov
print(ov.Core().available_devices)
# Expect: ['CPU']  at minimum;
# also 'GPU' if iGPU/dGPU + driver;
# also 'NPU' on Core Ultra with NPU driver.
```

### B.3 Convert ONNX to OpenVINO IR

```python
import openvino as ov
ov.save_model(
    ov.convert_model("experiments/exported_models/mobilenet_v3_small.onnx"),
    "experiments/exported_models/mobilenet_v3_small.xml",
)
```

You should now have two files: `mobilenet_v3_small.xml` (the graph) and `mobilenet_v3_small.bin` (the weights).

### B.4 Run inference on each device

```bash
# CPU
python src/inference/infer_openvino.py \
  --model experiments/exported_models/mobilenet_v3_small.xml \
  --image datasets/sample.jpg --device CPU

# iGPU (only if available)
python src/inference/infer_openvino.py \
  --model experiments/exported_models/mobilenet_v3_small.xml \
  --image datasets/sample.jpg --device GPU

# NPU (only if available; needs INT8-friendly model — see Ch 8 for quantization)
python src/inference/infer_openvino.py \
  --model experiments/exported_models/mobilenet_v3_small.xml \
  --image datasets/sample.jpg --device NPU

# AUTO (lets OpenVINO pick)
python src/inference/infer_openvino.py \
  --model experiments/exported_models/mobilenet_v3_small.xml \
  --image datasets/sample.jpg --device AUTO
```

### B.5 Benchmark and compare

| Runtime / Device | Precision | P50 ms | P95 ms | FPS estimate |
|---|---|---|---|---|
| ONNX Runtime CPU | FP32 | | | |
| OpenVINO CPU | FP32 | | | |
| OpenVINO iGPU (if available) | FP32 | | | |
| OpenVINO NPU (if available, after INT8 in Ch 8) | INT8 | | | |

### B.6 Reflection

In your lab report (max 1 page) answer:

1. How much speedup did OpenVINO CPU give over ONNX Runtime CPU on the same hardware?
2. If you have iGPU, was it faster than CPU for this model? Why might it not be?
3. Did top-1 prediction change between devices?
4. If you have NPU: what changed when you tried FP32 vs INT8? Did NPU even accept the FP32 model?

---

## Submission

Submit your lab report as `experiments/reports/lab_04_<your_name>.md`. Include:

- Which track you did (A, B, or both).
- The completed table for that track.
- Your 4 reflection answers (max 1 page total).
- A note on what hardware you tested on.

---

## Common pitfalls

- **TensorRT engine fails to load on a different device.** Engines are GPU-specific. Build on the device you will run on.
- **`available_devices` does not show `GPU` or `NPU`** even though the silicon is there. Driver not installed (Linux: `intel-opencl-icd`; NPU: Intel NPU driver).
- **OpenVINO NPU silently slow on FP32.** NPU is INT8-first; rerun after Chapter 8 with a quantized model.
- **Numbers vary by 30-50% between runs.** Set the power profile (`sudo nvpmodel`, `sudo jetson_clocks`) and avoid background load.
