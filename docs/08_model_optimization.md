# Chapter 8 — Model optimization: quantization, pruning, distillation

> **Goal:** Shrink and speed up models for edge deployment while controlling accuracy loss. By the end of this chapter you should be able to quantize a model to INT8 (dynamic and static), measure the size / latency / accuracy trade-off, and know when each technique helps and when it hurts.

This is where the "deployment" phase of the course turns into real wins. A model that runs at 20 FPS on a Jetson at FP32 can usually run at 50+ FPS at FP16 or INT8 — but only after the right combination of quantization, possibly pruning, and possibly distillation. The course teaches **measurement first**, **technique second**: every optimization is justified by numbers from `experiments/benchmark_results/`.

---

## 1. The three techniques

| Technique | What it does | Typical size reduction | Typical speedup | Typical accuracy loss |
|---|---|---|---|---|
| **Quantization** (FP16 / INT8) | Reduce bit-width of weights and/or activations | 2-4× | 1.5-4× (hardware-dependent) | 0-2% top-1 |
| **Pruning** | Zero out unimportant weights (structured or unstructured) | 1.5-5× | 1.1-3× (only with structured + sparse-capable runtime) | 0-3% top-1 |
| **Distillation** | Train a small "student" model to mimic a large "teacher" | depends on student size | depends on student size | 0-5% top-1, often better than training the student alone |

These can be combined: distill a small student, then quantize it, optionally also prune it.

The course covers **quantization in depth** (most impactful, easiest to apply) and gives an orientation to pruning and distillation.

---

## 2. The number representations

| dtype | Bits | Used for | Notes |
|---|---|---|---|
| FP32 | 32 | Training default | Baseline |
| FP16 | 16 | Inference on NVIDIA/AMD GPUs, Jetson, Apple NE | Almost free on supported hardware; usually <0.5% accuracy loss |
| BF16 | 16 | Training and inference on modern GPUs / NPUs | Same dynamic range as FP32 but lower precision |
| INT8 | 8 | Edge inference on CPU/NPU/TPU | 4× size reduction; needs calibration; accuracy loss depends on model |
| INT4 / W4A8 | 4 / mixed | Edge LLM (Ch 18); some CV | More accuracy loss; need recent runtimes (vLLM, llama.cpp, OpenVINO 2026) |

The course focuses on **FP16** (the easy first stop) and **INT8** (the standard for CPU/NPU/TPU edge inference).

---

## 3. Quantization paths

There are two practical ways to go INT8.

### 3.1 Dynamic quantization (weights only)

- Weights are quantized at *export* time.
- Activations are quantized on the fly at *runtime* (their ranges are estimated per-call).
- No calibration data needed.
- Easy and fast to apply.

**When to use:** as a first try, especially for **transformer / NLP-like models with Gemm-heavy layers**. For dense CV models on CPU, dynamic quantization sometimes slows down inference because the per-call activation quantization overhead exceeds the speedup from INT8 matmul. The course example below shows this.

### 3.2 Static quantization (weights + activations)

- Both weights and activations are quantized at export time.
- Activations need a **calibration dataset** so the quantizer can estimate their ranges.
- Larger speedup, especially on hardware with native INT8 (NPU, TPU, ARM dot-product instructions).
- Sometimes slightly more accuracy loss than dynamic.

**When to use:** for **CV models targeting CPU INT8 acceleration**, NPU (Intel Core Ultra), TPU (Coral), Jetson INT8 paths. This is the production path for any quantized deployment.

### 3.3 Quantization-aware training (QAT) — intro only

- The model is *trained* with simulated INT8 quantization in the forward pass (and FP32 gradients in the backward pass).
- Best accuracy retention, especially for INT8 of small or sensitive models.
- More work: you need the training pipeline and a labeled dataset.

In this course, QAT is introductory. The course defaults to **PTQ static** because it requires no retraining and is the realistic option for someone deploying an existing checkpoint.

---

## 4. The reference quantization helpers

`src/optimization/quantization.py` provides three functions:

```python
from src.optimization import (
    quantize_onnx_dynamic,
    quantize_onnx_static,
    compare_models,
    RandomImageCalibrationDataReader,
)
```

### 4.1 Dynamic INT8 in one call

```python
from pathlib import Path
from src.optimization import quantize_onnx_dynamic

src = Path("experiments/exported_models/mobilenet_v3_small.onnx")
dst = Path("experiments/exported_models/mobilenet_v3_small_int8_dyn.onnx")
quantize_onnx_dynamic(src, dst)
```

### 4.2 Static INT8 with calibration

```python
from src.optimization import quantize_onnx_static, RandomImageCalibrationDataReader

reader = RandomImageCalibrationDataReader(
    input_name="input", shape=(1, 3, 224, 224), num_samples=128,
)
quantize_onnx_static(src, dst, reader)
```

For real use, **replace `RandomImageCalibrationDataReader` with a reader that yields preprocessed images from your actual validation set**. 100-500 real samples is usually enough.

### 4.3 Compare across models

```python
from src.optimization import compare_models
from src.optimization.quantization import argmax_agreement

rows = compare_models(
    {"fp32": src, "int8_dyn": dyn_path, "int8_static": static_path},
    reference_label="fp32",
    metric_fn=argmax_agreement,
)
```

The result is a list of dicts with size, latency, FPS, and (vs reference) max abs diff and argmax agreement. The Chapter 4 notebook + the Chapter 4 assignment integrate this into a benchmark table.

---

## 5. The MobileNetV3-Small case study

This is the worked example used by `notebooks/chapter_04_quantization_ptq.ipynb`. Numbers below are from a 16-core CPU laptop running ONNX Runtime; your numbers will vary.

| Model | Size (MB) | P50 latency (ms) | Argmax agreement vs FP32 |
|---|---|---|---|
| MobileNetV3-Small FP32 (ONNX) | 9.71 | ~1.3 | 1.0 (reference) |
| INT8 dynamic | 2.59 | ~17 | high |
| INT8 static (real calibration) | 2.62 | ~3 | high |

Two important observations:

1. **The size win is real and consistent** — ~4× across both quantization paths.
2. **The latency win on small models on CPU is not automatic.** Dynamic INT8 *slows down* MobileNetV3-Small on this laptop because per-call activation quantization overhead dominates the INT8 matmul savings. Static INT8 is faster than dynamic, but still slower than the highly-optimized FP32 ORT path on this hardware.

Where INT8 wins clearly:
- On **NPU / TPU** (Intel Core Ultra NPU, Coral Edge TPU) where INT8 is the native datapath.
- On **Jetson with TensorRT INT8** when you have proper calibration.
- On **Raspberry Pi / ARM** with TFLite INT8.
- On **larger models** where the activation-quantization overhead is amortized.

The course's general rule: **size win is universal; speed win is hardware-dependent**. Always benchmark on the target device, not the dev machine.

---

## 6. Pruning (orientation)

Pruning zeros out weights deemed unimportant. Two flavors:

- **Unstructured pruning:** zero arbitrary scalars. High sparsity is possible but requires a sparse-aware runtime to translate to actual speedup. ONNX Runtime CPU does not.
- **Structured pruning:** zero whole channels / filters / heads. Produces a smaller dense model that any runtime can run faster.

Practical edge-AI guidance:
- Start with quantization. It is more impactful and easier.
- Add structured pruning when the model is still too big after quantization.
- Use frameworks like NVIDIA's NAS-style pruning, Hugging Face's `optimum`, or `torch.nn.utils.prune` for the basics.

The course does not implement pruning end-to-end; it is left as an extension exercise.

---

## 7. Knowledge distillation (orientation)

Distillation trains a small "student" model on the outputs (logits or features) of a large "teacher". The student learns a *smoother* target than hard labels alone provide, and often outperforms a student trained on hard labels from scratch.

Pipeline:

```
big teacher (e.g. ResNet-50)
       │
       ▼
small student (e.g. MobileNetV3-Small)
   trained on:  α · CE(student, hard_labels)
              + (1-α) · KL(soft_student_logits/T, soft_teacher_logits/T) * T²
```

Practical guidance for edge AI:
- Pick a student architecture that fits your latency budget.
- Distill with the teacher you already have.
- Quantize the distilled student.

The course does not implement distillation end-to-end either; the *concept* is what matters for the final project rubric.

---

## 8. Accuracy-vs-latency trade-off

The single useful chart for every optimization decision is **accuracy on the y-axis, latency (or model size) on the x-axis**. Every optimization moves a point on this chart. You ship the point that meets your accuracy floor at the smallest latency / size.

A useful workflow:

1. Establish the **FP32 baseline** (accuracy, latency, size).
2. Try **FP16** — usually free.
3. Try **INT8 dynamic** — almost free for transformer-style; sometimes worse for small CNNs.
4. Try **INT8 static (PTQ)** with real calibration data — the usual production choice.
5. If accuracy is still too low: try a **bigger student** or skip INT8 (stay at FP16).
6. If latency / size is still too high: try **distillation** or **structured pruning**.

Always re-benchmark after every step, save the numbers, and update the chart.

---

## 9. Common pitfalls

- **Quantizing with random calibration data.** The activation ranges are wrong; accuracy collapses. Use real validation samples.
- **Expecting an INT8 speedup on every device.** On CPU with small models, INT8 dynamic is often *slower*. Static is usually better but not guaranteed faster than well-tuned FP32 ORT.
- **Forgetting to compare argmax agreement.** "Looks like the loss is similar" is not a benchmark. Compute argmax agreement (or task metric) over a held-out set.
- **Comparing apples to oranges.** Quantizing to a TFLite INT8 model and benchmarking against a PyTorch FP32 model on a different machine produces meaningless deltas.
- **Pruning without a sparse-aware runtime.** Unstructured pruning that produces 50% sparse weights is just as slow as dense weights in ONNX Runtime — the FLOP count drops but the kernel doesn't know it.

---

## 10. What you should be able to do after this chapter

- Quantize an ONNX model with `quantize_onnx_dynamic` and `quantize_onnx_static`.
- Build a calibration data reader for your task's real validation samples.
- Compare FP32 / INT8 models on size, latency, and argmax agreement using `compare_models`.
- Decide whether quantization is appropriate for your target hardware and model class.
- Explain when pruning and distillation are worth investigating beyond quantization.

---

## 11. Files produced by this chapter

- `docs/08_model_optimization.md` — this file.
- `src/optimization/__init__.py`, `src/optimization/quantization.py` — quantization helpers.
- `notebooks/chapter_04_quantization_ptq.ipynb` — worked example with MobileNetV3-Small.
- `assignments/assignment_04_quantization.md` — graded assignment.
- `experiments/exported_models/*_int8_*.onnx` — quantized artifacts (produced by the notebook / scripts).
