# Chapter 7 — TensorRT, OpenVINO, and TFLite

> **Goal:** Survey the three production-grade runtimes you will see in real edge deployments — NVIDIA TensorRT, Intel OpenVINO, and Google TFLite (plus TFLite Micro) — and learn how to choose between them. By the end of this chapter you should know which runtime targets which hardware, what each one buys you over ONNX Runtime CPU, and how to convert your model to each.

ONNX Runtime is the universal default (Chapter 6). The runtimes in this chapter are **hardware-specific** and offer 2-10× more performance on their target devices — but at the cost of a longer toolchain and tighter version constraints.

This chapter is mostly a **decision-and-overview** chapter. Hands-on work happens in `labs/lab_04_tensorrt_or_openvino.md`, which you run only if you have matching hardware.

---

## 1. The three runtimes at a glance

| Runtime | Vendor | Target hardware | Format | Best for |
|---|---|---|---|---|
| **TensorRT** | NVIDIA | NVIDIA GPU / Jetson | `.engine` (compiled) | Real-time CV on Jetson, multi-stream, FP16/INT8 |
| **OpenVINO** | Intel | Intel CPU / iGPU / NPU | `.xml + .bin` (IR) | Intel laptops, NUCs, industrial PCs, Edge LLM with NPU |
| **TFLite** | Google | Mobile / ARM / Coral | `.tflite` (flatbuffer) | Android, iOS, Raspberry Pi, Coral Edge TPU |
| **TFLite Micro** | Google | Cortex-M / ESP32 / RISC-V MCUs | `.tflite` + generated C++ | TinyML (Ch 12) |

Each of these accepts an ONNX file (Ch 5) as input, but each runs its **own compilation step** to produce a hardware-specific artifact. That artifact is what you ship to the device — not the ONNX file directly.

---

## 2. TensorRT (NVIDIA)

### 2.1 What it is

TensorRT is NVIDIA's inference optimizer. It takes an ONNX model, fuses operators, picks the fastest kernels for your specific GPU, and produces a serialized `.engine` file. At inference time you load the engine and call `infer()`.

### 2.2 When to use

- You are deploying on a **Jetson** (Nano, Orin Nano, Orin NX, Orin AGX).
- You are deploying on a server / desktop NVIDIA GPU and want better throughput than ONNX Runtime + CUDA.
- You need **FP16 or INT8** with hardware acceleration.

### 2.3 Toolchain

```
PyTorch (.pt)
    ↓  Chapter 5
ONNX (.onnx)
    ↓  trtexec  /  polygraphy  /  Python TensorRT API
TensorRT engine (.engine / .plan)
    ↓
Python runtime (tensorrt + pycuda)  OR  ONNX Runtime TensorrtExecutionProvider
```

Simplest path: use `trtexec`, which ships with every JetPack and every desktop TensorRT install.

```bash
# FP32 baseline
trtexec --onnx=mobilenet_v3_small.onnx --saveEngine=mobilenet.engine

# FP16 — the typical Jetson default; usually no accuracy loss for vision models
trtexec --onnx=mobilenet_v3_small.onnx --saveEngine=mobilenet_fp16.engine --fp16

# INT8 — needs a calibration cache (Chapter 8 covers calibration)
trtexec --onnx=mobilenet_v3_small.onnx --saveEngine=mobilenet_int8.engine \
        --int8 --calib=calib.cache
```

### 2.4 Key constraints

- **Engine files are GPU-specific**. An engine built on an RTX 4090 does NOT run on a Jetson Orin. Always build on the target device, or use NVIDIA's cross-build flow.
- **First build is slow** (seconds to minutes) because TensorRT searches for the best kernel per layer. Cache the engine.
- **Dynamic shapes hurt perf.** Prefer fully static engines per resolution if you can.
- **Op coverage is limited.** Some custom layers (e.g. non-standard NMS) need a plugin.

### 2.5 Reference script

`src/inference/infer_tensorrt.py` (template) shows how to load an engine and run inference. It only runs on a machine with TensorRT installed (typically a Jetson).

---

## 3. OpenVINO (Intel)

### 3.1 What it is

OpenVINO is Intel's inference toolkit. It takes an ONNX (or TF / PaddlePaddle / PyTorch) model, converts it to an Intermediate Representation (IR — `.xml` + `.bin`), and runs it on Intel CPU, iGPU, or NPU through a single Python API. The same compiled model can run on any of the three devices by changing one string.

### 3.2 When to use

- You deploy on **Intel hardware** (NUCs, industrial PCs, x86 edge boxes, laptops with Iris Xe iGPU).
- You want to use the **Intel NPU** (Core Ultra "Meteor Lake" and later).
- You need **Windows-friendly** deployment.
- You want **heterogeneous** execution (e.g. CV on iGPU + LLM attention on NPU).

### 3.3 Toolchain

```
PyTorch / ONNX / TF model
    ↓  openvino.convert_model(...)  /  ovc CLI
OpenVINO IR (.xml + .bin)
    ↓  optional: nncf.quantize(...) for INT8
Quantized IR
    ↓
core.compile_model(IR, device_name="CPU"|"GPU"|"NPU"|"AUTO")
```

Simplest path:

```python
import openvino as ov
ov.save_model(ov.convert_model("model.onnx"), "model.xml")

core = ov.Core()
compiled = core.compile_model("model.xml", device_name="AUTO")  # picks best
infer = compiled.create_infer_request()
result = infer.infer({0: x})
```

`device_name="AUTO"` is convenient. For repeatable benchmarks, pin the device explicitly: `"CPU"`, `"GPU"`, or `"NPU"`.

### 3.4 Key constraints

- **NPU is INT8-first.** Run quantization (Chapter 8) before targeting NPU; FP32 either won't run or runs slowly.
- **Op support per device varies.** Always check `available_devices` and try `AUTO` first.
- **GPU driver / NPU driver** must be installed at OS level; pip install of OpenVINO is not enough.
- **OpenVINO 2026 brought big NPU improvements** for LLM workloads (3.8× throughput on 7B-class attention vs GPU-only).

### 3.5 Reference script

`src/inference/infer_openvino.py` (template) loads an IR or ONNX and runs through OpenVINO. Tested only on machines with OpenVINO installed.

---

## 4. TFLite (mobile / edge / Coral)

### 4.1 What it is

TFLite is Google's mobile/edge inference engine. It uses a flatbuffer `.tflite` file. It originated for TensorFlow models, but you can convert from ONNX via `tf2onnx` + `tensorflow` or via `onnx2tf`.

### 4.2 When to use

- **Mobile** (Android, iOS).
- **Raspberry Pi** with INT8 models — `tflite-runtime` is small (a few MB) compared to PyTorch / ONNX Runtime.
- **Google Coral Edge TPU** — only TFLite Edge TPU models run on Coral.
- **TFLite Micro** (Ch 12) for Cortex-M microcontrollers.

### 4.3 Toolchain

PyTorch → ONNX → TFLite is the most common path for this course:

```bash
# Convert ONNX → TF SavedModel → TFLite
# (Two-step because there is no direct ONNX → TFLite tool maintained by either vendor)
onnx2tf -i model.onnx -o model_tf -ois "input:1,3,224,224"
# Inside model_tf/ you get a SavedModel; convert to TFLite:
python -c "import tensorflow as tf; \
    c = tf.lite.TFLiteConverter.from_saved_model('model_tf'); \
    open('model.tflite','wb').write(c.convert())"
```

For Coral, add the Edge TPU compiler:

```bash
edgetpu_compiler model_int8.tflite   # produces model_int8_edgetpu.tflite
```

For TFLite Micro (Chapter 12), the converter is the same; you then run `xxd -i model.tflite > model.cc` to generate a C++ array.

### 4.4 Key constraints

- **INT8 is mandatory** for Coral Edge TPU and TFLite Micro.
- **Operator support is limited** on Edge TPU and Micro — not every ONNX model converts cleanly.
- The **two-step ONNX → TF → TFLite** path is brittle; for production-targeting TFLite, you may prefer to train in TensorFlow / Keras and skip ONNX entirely.

### 4.5 Course coverage

TFLite hands-on lives in Chapter 11 (sensor / IoT) and Chapter 12 (TinyML), where the operator set is naturally small enough that the conversion just works.

---

## 5. Decision tree: which runtime?

```
target hardware?
│
├── NVIDIA Jetson / RTX GPU?
│       └── TensorRT  (FP16 default, INT8 if you can calibrate)
│
├── Intel x86 / NUC / Core Ultra / Arc?
│       └── OpenVINO  (try AUTO; quantize to INT8 for NPU)
│
├── Apple Silicon / iPhone / iPad?
│       └── Core ML  (out of course scope; ORT CoreMLExecutionProvider works)
│
├── Raspberry Pi (no accelerator)?
│       └── TFLite INT8  or  ONNX Runtime CPU
│
├── Raspberry Pi + Coral?
│       └── TFLite Edge TPU INT8
│
├── Microcontroller (Cortex-M, ESP32)?
│       └── TFLite Micro INT8 (Chapter 12)
│
├── Hailo / specialized NPU?
│       └── Vendor SDK + ONNX  (Hailo runtime, etc.)
│
└── Anything else / unsure?
        └── ONNX Runtime CPU (always works, ship it, optimize later)
```

---

## 6. Comparison: same model, four runtimes

The course's worked example uses MobileNetV3-Small (224×224 input, ImageNet). The numbers below are typical; your own benchmarks will vary by hardware.

| Runtime | Hardware | P50 ms | Notes |
|---|---|---|---|
| PyTorch CPU | Laptop CPU (16 cores) | ~5 ms | The "naïve" baseline |
| ONNX Runtime CPU | Laptop CPU (16 cores) | ~1.5-2 ms | 2-3× faster, no hardware change |
| ONNX Runtime + CUDA | Laptop NVIDIA GPU | ~1 ms | Below 1 ms is rare for batch=1 (kernel launch dominates) |
| TensorRT FP16 | Jetson Orin Nano (25W) | ~3-5 ms | Plus 60+ FPS end-to-end with camera capture |
| OpenVINO INT8 | Intel Core Ultra NPU | ~2-3 ms | Best perf/W on Intel hardware |
| TFLite INT8 CPU | Raspberry Pi 5 | ~10-20 ms | Realistic for SBC without accelerator |
| TFLite Edge TPU | Pi 5 + Coral USB | ~5-8 ms | INT8 only |

Lessons:
- The first **2-3× speedup is "free"** — same hardware, just ORT instead of PyTorch.
- Beyond that, **runtime + hardware specialization** is what wins.
- The choice is not "which runtime is fastest" but "which runtime fits the hardware I am shipping on".

---

## 7. The reference inference scripts

| Script | Hardware required | Runs at Level 1? |
|---|---|---|
| `src/inference/infer_pytorch.py` | Any | ✓ |
| `src/inference/infer_onnxruntime.py` | Any | ✓ |
| `src/inference/infer_tensorrt.py` | NVIDIA GPU with TensorRT | only on Jetson / desktop CUDA |
| `src/inference/infer_openvino.py` | Intel CPU / iGPU / NPU | runs on Intel CPU at Level 1 |

The TensorRT and OpenVINO scripts include fallback messages if the runtime is not installed — they will not crash, just print "install the runtime first."

---

## 8. The lab

`labs/lab_04_tensorrt_or_openvino.md` walks through the hands-on for the runtime that matches your hardware:

- If you have a Jetson → TensorRT path.
- If you have an Intel iGPU / NPU → OpenVINO path.
- If you have neither → read both, run the OpenVINO path on Intel CPU (it works without iGPU).

---

## 9. What you should be able to do after this chapter

- Tell which runtime fits which hardware.
- Convert an ONNX model to a TensorRT engine and to an OpenVINO IR.
- Read the comparison table and explain what the column differences mean.
- Decide which runtime to target for your final project, and justify it.

---

## 10. Files produced by this chapter

- `docs/07_tensorrt_openvino_tflite.md` — this file.
- `src/inference/infer_tensorrt.py` — template TensorRT inference script.
- `src/inference/infer_openvino.py` — template OpenVINO inference script.
- `labs/lab_04_tensorrt_or_openvino.md` — guided lab (hardware-dependent).
