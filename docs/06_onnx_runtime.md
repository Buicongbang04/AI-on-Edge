# Chapter 6 — ONNX Runtime for edge inference

> **Goal:** Run an exported ONNX model using ONNX Runtime, understand what an **execution provider** is, and benchmark CPU vs GPU vs (optionally) hardware-specific providers. By the end of this chapter you should be able to swap providers with a one-line change and reason about which one fits your hardware.

ONNX Runtime (ORT) is the simplest production-grade inference engine for ONNX models. It runs on Linux, Windows, macOS, Android, iOS, and embedded Linux, supports many hardware backends, and has stable Python and C++ APIs.

This is the path the course recommends as your default inference runtime — until you have a specific reason to use TensorRT (Jetson), OpenVINO (Intel hardware), or TFLite (microcontroller / mobile).

---

## 1. Why ONNX Runtime first

| Property | Why it matters here |
|---|---|
| Cross-platform | Same model runs on your laptop, a Pi, a Jetson (CPU mode), or a NUC |
| Stable API | Python and C++ APIs have changed little since 2020 |
| Hardware-agnostic until you opt in | You pick the *execution provider*, model file is unchanged |
| Fast on CPU | Often 2-5× faster than raw PyTorch CPU for small models |
| Tooling | `onnxruntime-tools`, profiling, quantization (Ch 8) |

ORT is the bridge: you can deploy with ORT *today*, then drop in a faster runtime (TensorRT, OpenVINO) on the same ONNX file *later*.

---

## 2. Execution providers

An **execution provider** (EP) is ORT's plug-in for a specific hardware/runtime backend. When you create a session you pass a *list* of providers and ORT picks the first one each operator supports, falling back down the list for unsupported ops.

| Provider | Where it runs | Install hint |
|---|---|---|
| `CPUExecutionProvider` | Any CPU (always available) | bundled with `onnxruntime` |
| `CUDAExecutionProvider` | NVIDIA GPU with CUDA | `pip install onnxruntime-gpu` (replaces CPU package) |
| `TensorrtExecutionProvider` | NVIDIA GPU / Jetson via TensorRT | with onnxruntime-gpu + matching TensorRT install |
| `OpenVINOExecutionProvider` | Intel CPU / iGPU / NPU | with OpenVINO toolkit; specific wheel |
| `CoreMLExecutionProvider` | Apple Silicon / iOS | macOS wheels |
| `DmlExecutionProvider` | Windows DirectML (Intel / AMD / NVIDIA GPUs on Windows) | Windows wheels |
| `XnnpackExecutionProvider` | Mobile-class CPU (ARM) | mobile-focused wheels |
| `NnapiExecutionProvider` | Android NNAPI | Android NDK builds |

Check what you actually have:

```python
import onnxruntime as ort
print(ort.get_available_providers())
# e.g. ['CUDAExecutionProvider', 'CPUExecutionProvider']
#   or ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
#   or ['CPUExecutionProvider', 'AzureExecutionProvider']   ← typical pip install on a CPU-only box
```

If you do not see the provider you need, **install the right wheel** (see table) — having ORT alone is not enough.

---

## 3. Minimum viable inference

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(
    "model.onnx",
    providers=["CPUExecutionProvider"],
)
in_name = sess.get_inputs()[0].name
x = np.random.randn(1, 3, 224, 224).astype(np.float32)

(logits,) = sess.run(None, {in_name: x})
print(logits.shape, logits.dtype)
```

That is the entire API. Real code adds preprocessing and postprocessing (Chapter 9), and benchmarking (Chapter 4).

---

## 4. Session options worth knowing

```python
opts = ort.SessionOptions()

# Graph optimizations (default is ENABLE_ALL on most builds).
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Threading on CPU. 0 = let ORT pick from physical cores.
opts.intra_op_num_threads = 0   # ops within a single forward pass
opts.inter_op_num_threads = 0   # parallelism across independent subgraphs (rare for vision)

# Optional: save the optimized graph to disk for faster cold starts later
opts.optimized_model_filepath = "model_optimized.onnx"

# Optional: enable profiling. Writes a Chrome trace JSON.
opts.enable_profiling = True

sess = ort.InferenceSession("model.onnx", sess_options=opts, providers=["CPUExecutionProvider"])
```

For edge deployment, two of these matter most:

- **`intra_op_num_threads`** — on a Pi 5 with 4 cores, 4 is right; on a Jetson Orin Nano (6-core CPU), 6 is right. The default (0) usually picks well, but on systems with E-cores or heavy background load, set it explicitly.
- **`graph_optimization_level`** — leave it at ENABLE_ALL. Disable only for debugging.

---

## 5. Input / output binding

For most use cases, `sess.run(None, {in_name: x})` is enough. Two advanced patterns:

### 5.1 Multiple inputs

```python
inputs = {sess.get_inputs()[i].name: tensors[i] for i in range(len(tensors))}
outputs = sess.run(None, inputs)
```

### 5.2 IO binding (zero-copy GPU input)

When using `CUDAExecutionProvider`, you can avoid CPU↔GPU copies by binding an already-on-GPU tensor:

```python
io = sess.io_binding()
io.bind_input("input", device_type="cuda", device_id=0, element_type=np.float32,
              shape=(1, 3, 224, 224), buffer_ptr=gpu_tensor.data_ptr())
io.bind_output("logits", device_type="cuda")
sess.run_with_iobinding(io)
```

This matters most when chaining several models (e.g. detector → tracker) on the same GPU. For a single batch=1 inference it usually does not change much.

---

## 6. Benchmarking ONNX Runtime

Reuse `src.benchmark` from Chapter 4:

```python
import numpy as np
import onnxruntime as ort
from src.benchmark import bench_full, format_report

sess = ort.InferenceSession("experiments/exported_models/mobilenet_v3_small.onnx",
                            providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
x = np.random.randn(1, 3, 224, 224).astype(np.float32)

def step():
    return sess.run(None, {in_name: x})

report = bench_full(step, name="mnv3s-onnxrt-cpu-fp32", device="cpu",
                    extras={"runtime": "onnxruntime", "provider": "CPUExecutionProvider"})
print(format_report(report))
```

Compare the report to the PyTorch one from Chapter 4 — on CPU you should typically see ORT come in **2-5× faster** for small models like MobileNetV3-Small. Larger models often see less of a gap.

---

## 7. CPU vs CUDA vs TensorRT (when you have CUDA)

If you have an NVIDIA GPU and `onnxruntime-gpu` installed:

```python
# CUDA
sess_cuda = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# TensorRT (requires TensorRT installed; first run compiles an engine and caches it)
sess_trt = ort.InferenceSession(model_path, providers=[
    ("TensorrtExecutionProvider", {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "experiments/exported_models/trt_cache",
    }),
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
])
```

Two important facts:

- The **first run** of the TensorRT provider builds the engine — it can take minutes. Subsequent runs read from cache.
- `providers` is a *fallback list*: TensorRT first, then CUDA, then CPU. Any op TensorRT cannot handle silently falls back.

Benchmark all three and report the table. On a Jetson, you should see TensorRT FP16 win clearly. On a desktop GPU, CUDA and TensorRT are usually close on small models.

---

## 8. The reference script

`src/inference/infer_onnxruntime.py` mirrors `infer_pytorch.py` and runs the same kind of inference on the ONNX file:

```bash
# Default: CPU
python src/inference/infer_onnxruntime.py \
    --model experiments/exported_models/mobilenet_v3_small.onnx \
    --image datasets/sample.jpg

# CUDA, if onnxruntime-gpu is installed
python src/inference/infer_onnxruntime.py \
    --model experiments/exported_models/mobilenet_v3_small.onnx \
    --image datasets/sample.jpg --provider CUDAExecutionProvider
```

Sample output:

```
loading experiments/exported_models/mobilenet_v3_small.onnx (provider=CPUExecutionProvider) ...
warming up for 5 iteration(s) on sample.jpg ...

running inference on 1 image(s):

[sample.jpg]  latency=1.68 ms
   21.64%  candle
   17.40%  jack-o'-lantern
    ...
```

The predictions and top-5 ordering should match the PyTorch output exactly (we already validated this in Chapter 5).

---

## 9. Common pitfalls

- **`AzureExecutionProvider` shows up in `get_available_providers()` but you wanted CUDA.** That is the default `pip install onnxruntime` (CPU-only) result. To get CUDA, `pip uninstall onnxruntime` and `pip install onnxruntime-gpu`.
- **`InvalidArgument: input … shape mismatch`** at session.run — you exported a static-shape model but are feeding a different shape. Re-export with `--dynamic` or feed the trained shape.
- **`Onnxruntime: Failed to load library libcudnn.so.x`** on Jetson / WSL — version mismatch. ORT-GPU is pinned to specific CUDA/cuDNN versions; check the table on the ORT GitHub.
- **TensorRT provider silently falls back to CUDA for every op.** Inspect with `OrtLogger`, or compile an engine via `trtexec --onnx=...` and watch for `[W]` warnings about unsupported ops.
- **Single thread set to `1` on a multi-core Pi.** ORT defaults to picking the right count, but if you set `intra_op_num_threads=1` you cap performance at 1/N of the device.

---

## 10. What you should be able to do after this chapter

- Load and run an ONNX model with `onnxruntime`.
- Read `get_available_providers()` and pick the right EP for your hardware.
- Configure session options for threading and graph optimization.
- Benchmark ORT with the Chapter 4 toolkit and produce a CPU-vs-GPU comparison.
- Recognize the common provider / shape / version error messages.

---

## 11. Files produced by this chapter

- `docs/06_onnx_runtime.md` — this file.
- `src/inference/infer_onnxruntime.py` — the reference inference script.
- `notebooks/chapter_03_onnxruntime_inference.ipynb` — chapter notebook (CPU + GPU comparison).
