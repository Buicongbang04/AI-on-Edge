# Chapter 5 — Export model: PyTorch → ONNX

> **Goal:** Understand ONNX as the *intermediate representation* that connects training (often PyTorch / TensorFlow) to many runtimes (ONNX Runtime, TensorRT, OpenVINO, TFLite via tflite-converter, Core ML). By the end of this chapter you should be able to export a PyTorch classifier to ONNX, validate it, and confirm its output matches the PyTorch original.

This is the first deployment step: getting your model out of the training framework. Almost every later optimization (Ch 6-8) and runtime (Ch 7) assumes the model is already in ONNX.

---

## 1. Why export?

PyTorch is great for training. It is *fine* for inference. But:

- PyTorch has a large Python footprint.
- It is hard to ship on a microcontroller, on iOS, or to a runtime written in C++.
- It does not auto-optimize for INT8, TensorRT engines, or Intel NPUs.

ONNX is a portable graph format. Once a model is in ONNX, you can:

- Load it in **ONNX Runtime** (Ch 6) for CPU/GPU/NPU inference.
- Compile it to a **TensorRT** engine (Ch 7) for NVIDIA Jetson.
- Convert it to **OpenVINO** IR (Ch 7) for Intel hardware.
- Convert it to **TFLite** (Ch 7) for mobile/edge.
- Ship it to **Core ML** for iOS / macOS.

One export, many runtimes.

---

## 2. ONNX in one paragraph

ONNX (Open Neural Network Exchange) is a serialization format for ML compute graphs. It defines:

- **Operators** (Conv, Gemm, BatchNorm, Relu, etc.) — versioned per **opset**.
- **Tensors** with named inputs and outputs, typed and shaped.
- An **IR version** that controls the file format.

A `.onnx` file is a protobuf serialization of a graph + weights. You can inspect it with `onnx.checker`, visualize it with [Netron](https://netron.app/), or load it in any compatible runtime.

The two things that bite people:

1. **Opset version.** Different runtimes support different opsets. ONNX Runtime 1.17+ supports opset 20; older TensorRT might cap at opset 17. Default to **opset 17** for broad compatibility.
2. **Static vs dynamic shapes.** Many runtimes want fixed input shapes for maximum performance. Exporting with `dynamic_axes` is convenient, but may slow down some runtimes (especially TensorRT) compared to a fully static graph.

---

## 3. TorchScript vs torch.onnx vs torch.export

Three PyTorch paths to portable models:

| Path | Output | Used when |
|---|---|---|
| **TorchScript** (`torch.jit.trace` / `torch.jit.script`) | `.pt` script module | Keeping inside PyTorch / LibTorch C++ |
| **torch.onnx.export** (legacy, TorchScript-based) | `.onnx` (single file) | This course's default — broad compatibility, simple |
| **torch.onnx.export** with `dynamo=True` (PyTorch 2.9+ default) | `.onnx` (+ optional `.onnx.data` for weights) | Newer; uses `torch.export` graph; handles control flow better |

The course uses the **legacy TorchScript-based** exporter (`dynamo=False`) because:

- It produces a **single file** by default (no `.onnx.data` sidecar), which is easier for learners.
- It supports the broadest range of older runtimes (TensorRT on JetPack 5.x, OpenVINO 2023.x).
- Behavior is stable across PyTorch 2.x.

You will see a deprecation warning. It is safe to ignore in this course; for production code targeting PyTorch 3.x, you would migrate to the dynamo exporter.

---

## 4. Minimum viable export

```python
import torch
from torchvision import models

# 1. Load model in eval mode
model = models.mobilenet_v3_small(weights="DEFAULT").eval()

# 2. Create a dummy input with the SAME shape and dtype the model expects
dummy = torch.randn(1, 3, 224, 224)

# 3. Export
torch.onnx.export(
    model,
    dummy,
    "mobilenet_v3_small.onnx",
    input_names=["input"],          # name your inputs
    output_names=["logits"],        # and outputs
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,                   # legacy single-file exporter
)
```

Three things to notice:

1. **The dummy must match the real input shape and dtype.** The exporter traces the forward pass with this input; if the real input is FP16 but you trace with FP32, the resulting graph is FP32.
2. **`do_constant_folding=True`** asks the exporter to fold constant subgraphs (e.g. parameter reshapes) into pre-computed constants. Always on for deployment.
3. **`opset_version=17`** is the safe default for 2024-2026 runtimes.

---

## 5. Static vs dynamic shapes

By default, the exported graph has the exact shapes from the dummy input. That is great for performance (some runtimes specialize kernels per shape) but bad if you want to switch batch sizes at inference.

To allow variable shapes, pass `dynamic_axes`:

```python
torch.onnx.export(
    model, dummy, "mobilenet_v3_small_dyn.onnx",
    input_names=["input"], output_names=["logits"],
    dynamic_axes={
        "input":  {0: "batch", 2: "height", 3: "width"},
        "logits": {0: "batch"},
    },
    opset_version=17, dynamo=False,
)
```

Trade-offs:
- **Pro:** one ONNX file works for batch=1 and batch=N, and for variable resolution.
- **Con:** TensorRT in particular benefits a lot from fully static shapes (sometimes 1.5-3× faster). If you target Jetson, export a *static* model for each shape you care about.

A common pattern: export both. Keep the static one for benchmarking, the dynamic one for general use.

---

## 6. Validating the export

Two validations to run after every export:

### 6.1 `onnx.checker` — does the file parse?

```python
import onnx
model = onnx.load("mobilenet_v3_small.onnx")
onnx.checker.check_model(model)
print("ir_version:", model.ir_version, "opset:", model.opset_import[0].version)
```

If `check_model` raises, the file is malformed — usually opset or shape inconsistency. Re-export, often with a lower opset.

### 6.2 PyTorch vs ONNX Runtime output diff

The serious check: does ONNX produce the same numbers as PyTorch?

```python
import numpy as np, torch
import onnxruntime as ort
from torchvision import models

model = models.mobilenet_v3_small(weights="DEFAULT").eval()
x = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    y_torch = model(x).numpy()

sess = ort.InferenceSession("mobilenet_v3_small.onnx", providers=["CPUExecutionProvider"])
y_ort = sess.run(None, {"input": x.numpy()})[0]

print("max abs diff:", np.abs(y_torch - y_ort).max())  # expect ~1e-5
print("cosine:", (y_torch.flatten() @ y_ort.flatten()) / (np.linalg.norm(y_torch) * np.linalg.norm(y_ort)))
print("same argmax:", np.argmax(y_torch, 1)[0] == np.argmax(y_ort, 1)[0])
```

Acceptable thresholds for FP32 models:

| Check | Pass criterion |
|---|---|
| `max abs diff` | < 1e-4 |
| `cosine` | > 0.9999 |
| `same argmax` | always |

If you see a much larger diff, common causes:
- BatchNorm in training mode (forgot `model.eval()`).
- Custom op that ONNX implements differently.
- Non-deterministic operator (e.g. some `Interpolate` modes).

For FP16 or INT8 exports the thresholds relax (e.g. `max abs diff < 1e-1`, cosine > 0.999). At INT8 even argmax can disagree on a few items in a batch — that is normal as long as task metrics hold.

---

## 7. Common export errors and how to read them

| Error | Likely cause | Fix |
|---|---|---|
| `RuntimeError: Exporting the operator <X> ...` | Op not in the chosen opset | Raise `opset_version` or rewrite the layer |
| `Tracing fails because of dynamic control flow` | `if x.shape[0] > 5: ...` inside `forward` | Refactor; or use `dynamo=True` |
| `Output mismatch (large abs diff)` | `model.train()` instead of `model.eval()`, or stochastic op | Force eval; check Dropout / BatchNorm |
| `dynamic shape ignored` | Runtime built for static shapes | Re-export static, or use a runtime that supports dynamic |
| `Cannot find input name 'input.1'` at inference | Did not pass `input_names`; PyTorch defaults to numeric names | Always pass `input_names`, `output_names` |
| `.onnx.data` sidecar appears | Using new dynamo exporter | Pass `dynamo=False` to keep one file |

---

## 8. The reference script

`src/export/export_onnx.py` wraps the export, the `onnx.checker` validation, and the PyTorch-vs-ONNX numerical diff into one CLI.

```bash
# Default static export of MobileNetV3-Small
python src/export/export_onnx.py --model mobilenet_v3_small

# Dynamic axes
python src/export/export_onnx.py --model mobilenet_v3_small --dynamic

# Bigger model
python src/export/export_onnx.py --model resnet50 --opset 17
```

Output:

```
exporting mobilenet_v3_small → experiments/exported_models/mobilenet_v3_small.onnx
  saved experiments/exported_models/mobilenet_v3_small.onnx  size=9.71 MB
  ONNX check OK. ir_version=8 opset=17
  max|diff|=0.000009  cosine=1.000000  same_argmax=1  OK
```

The exported file lands in `experiments/exported_models/`. Every later chapter (6, 7, 8) reads from that directory.

---

## 9. Inspecting the exported graph

Two tools are worth knowing:

- **Netron** (https://netron.app) — drag-and-drop visualizer for ONNX, TFLite, Core ML. Indispensable for debugging "why is this layer slow?".
- **`onnx.helper.printable_graph(model.graph)`** — quick text dump for grep / diff.

```python
import onnx
m = onnx.load("mobilenet_v3_small.onnx")
print(onnx.helper.printable_graph(m.graph)[:2000])
```

Use Netron whenever something is slower than you expected on Jetson / OpenVINO — the issue is often an op (e.g. a leftover `Cast`, an unfused `Add` after Conv) that the runtime cannot optimize.

---

## 10. What you should be able to do after this chapter

- Export a PyTorch torchvision classifier to ONNX with `src/export/export_onnx.py`.
- Validate the ONNX file with `onnx.checker`.
- Confirm the ONNX output matches PyTorch within FP32 tolerance.
- Decide between static and dynamic export for your target runtime.
- Read a common export error message and act on it.

---

## 11. Files produced by this chapter

- `docs/05_pytorch_to_onnx.md` — this file.
- `src/export/export_onnx.py` — the export + validation CLI.
- `notebooks/chapter_02_pytorch_to_onnx.ipynb` — chapter notebook.
- `labs/lab_02_export_pytorch_to_onnx.ipynb` — guided lab.
- `experiments/exported_models/<model>.onnx` — exported artifacts (produced by the CLI).
