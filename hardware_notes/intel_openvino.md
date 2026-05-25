# Hardware notes: Intel CPU / iGPU / NPU + OpenVINO

OpenVINO is Intel's inference toolkit. It runs models on Intel CPUs, Iris Xe iGPUs, and (on Core Ultra / Meteor Lake / Lunar Lake / Arrow Lake) Intel NPUs — the same model file is dispatched to whichever device you choose. As of OpenVINO 2026, the NPU path is mature enough that 7B-class LLMs run with ~3.8× throughput improvement vs the iGPU-only path on the same machine.

This makes Intel hardware the natural choice for two scenarios in this course:

- Edge inference on **x86 industrial PCs and NUC-class devices** that ship with Intel SoCs.
- **Heterogeneous workloads** (CV + LLM, multi-model, mixed-precision) where you want to split work across CPU, iGPU, and NPU.

---

## Devices and capability

| Device class | Hardware | What OpenVINO targets |
|---|---|---|
| Generic Intel CPU | Any 6th gen+ Core, Xeon, Atom | CPU plugin |
| iGPU-equipped (Iris Xe, Arc) | 11th gen+ Core, Core Ultra | GPU plugin (auto-uses iGPU) |
| NPU-equipped (Core Ultra "Meteor Lake" and later) | Core Ultra 100/200, Lunar Lake, Arrow Lake | NPU plugin |
| Discrete Intel Arc GPU | A380/A750/A770 | GPU plugin |

A typical 2026 NUC-class machine (Core Ultra 7) has all three: CPU + iGPU + NPU. OpenVINO can run the same model on any of them by changing one string (`"CPU"`, `"GPU"`, `"NPU"`).

---

## When to use OpenVINO

Good fit:
- You are deploying on Intel hardware (the user runs Windows / Linux on an x86 PC).
- You want one model file to run on CPU when no accelerator, then accelerate when one shows up.
- Heterogeneous workload — CV on iGPU + LLM on NPU is a real production pattern in 2026.
- You need a Windows-friendly deployment path (Windows + OpenVINO is well-supported).

Not the best fit:
- NVIDIA Jetson — use TensorRT instead.
- Microcontroller — use TFLite Micro instead.
- Pure Apple Silicon — use Core ML instead.

---

## Install (Linux example)

OpenVINO ships as a pip package since 2023; this is the simplest path.

```bash
pip install openvino openvino-dev

# Verify
python -c "import openvino as ov; print(ov.__version__); print(ov.Core().available_devices)"
```

`available_devices` should print at least `['CPU']`, plus `'GPU'` if an Iris Xe / Arc is detected, plus `'NPU'` on Core Ultra. If you do not see `GPU` or `NPU` when you expect to, install the Intel compute driver:

```bash
# Intel Compute Runtime (for iGPU/dGPU OpenCL support)
sudo apt install -y intel-opencl-icd
# NPU driver (Linux) — see Intel's docs; the driver package is named `intel-driver-compiler-npu`
```

For Windows, the OpenVINO installer or the pip package "just works"; the iGPU and NPU drivers ship with the Intel Graphics Driver bundle.

---

## The OpenVINO workflow

```
PyTorch / ONNX / TensorFlow model
            ↓
OpenVINO IR (.xml + .bin)        ← optional but recommended for production
            ↓
ov.Core().compile_model(IR, device="CPU"|"GPU"|"NPU"|"AUTO")
            ↓
infer_request.infer({input_name: x})  → output
```

You do not strictly need to convert to IR — OpenVINO can also load `.onnx` directly. But converting to IR ahead of time enables INT8 quantization with the OpenVINO POT/NNCF tools and gives faster cold-start.

### Conversion example

```python
import openvino as ov
import openvino.runtime as ovr

# Convert ONNX to OpenVINO IR
ov_model = ov.convert_model("model.onnx")
ov.save_model(ov_model, "model.xml")

# Compile for a device
core = ovr.Core()
compiled = core.compile_model("model.xml", device_name="AUTO")  # picks best device
infer = compiled.create_infer_request()

# Inference
result = infer.infer({0: image_tensor})
```

`device_name="AUTO"` lets OpenVINO pick GPU > NPU > CPU based on availability and model compatibility — a good default for portable code.

---

## Quantization with OpenVINO

NNCF (Neural Network Compression Framework) provides post-training INT8 quantization for OpenVINO IR:

```python
import nncf
import openvino as ov

dataset = nncf.Dataset(calibration_loader, transform_fn)
quantized = nncf.quantize(
    model=ov.convert_model("model.onnx"),
    calibration_dataset=dataset,
)
ov.save_model(quantized, "model_int8.xml")
```

This is the cleanest INT8 path on Intel hardware. The same compiled IR runs on CPU, iGPU, or NPU — with NPU usually giving the best perf/W for INT8 workloads.

---

## Heterogeneous execution

OpenVINO can split a model across devices automatically:

```python
compiled = core.compile_model("model.xml", device_name="HETERO:NPU,GPU,CPU")
```

In practice, on Core Ultra:
- CV detectors (YOLO-class) run great on iGPU.
- LLMs (7B-class) benefit from NPU offload of attention layers.
- Postprocessing usually stays on CPU.

The Intel 2026 NPU release notes report ~3.8× throughput on 7B LLMs when attention layers are offloaded to NPU vs GPU-only. For mixed workloads (camera + LLM), this is the path to investigate.

---

## Performance expectations

Approximate end-to-end FPS on Intel Core Ultra 7 (NUC-class) with OpenVINO 2026, INT8:

| Workload | Device | FPS |
|---|---|---|
| YOLOv8n 640×640 | iGPU | 60-100 |
| YOLOv8n 640×640 | NPU | 80-120 |
| MobileNetV3 classifier | NPU | 500+ |
| 7B LLM, 4K context, decode | NPU + iGPU hetero | ~25-40 tok/s |

These are rough numbers — produce your own with Chapter 4's benchmark template.

---

## Power and thermals

- Intel NUC under sustained 30 W: needs the stock fan (it has one).
- Core Ultra NPU is designed for low power; sustained NPU-only workloads draw <5 W incremental over idle.
- Fanless industrial PCs ship with conservative TDPs; check the device's documented sustained TDP before counting on max performance.

---

## Common pitfalls

- **No iGPU/NPU shown in `available_devices`.** Driver is not installed or you are inside a container that does not expose `/dev/dri` (Linux). On bare metal: install the Intel compute driver. In Docker: mount `/dev/dri` and run with the `render` group.
- **Trying NPU with operators it does not support.** OpenVINO will silently fall back to CPU for unsupported ops. Use the `AUTO` device or check the OpenVINO compatibility matrix.
- **Forgetting to quantize for NPU.** NPU is INT8-first; running FP32 on NPU is either unsupported or much slower than iGPU.
- **Comparing OpenVINO numbers to TensorRT numbers across machines.** Different hardware, different runtimes. Comparing only makes sense per-device.

---

## Where to go next

- Chapter 6 — ONNX Runtime with `OpenVINOExecutionProvider` if you want to keep the ONNX Runtime API.
- Chapter 7 — OpenVINO lab (`labs/lab_04_tensorrt_or_openvino.md`).
- Chapter 8 — INT8 quantization with NNCF on Intel hardware.
- Chapter 18 — Edge LLM on Core Ultra NPU.
