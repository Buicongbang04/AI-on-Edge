# Assignment 4 — INT8 quantization trade-off

**Chapter:** 8 — Model optimization
**Type:** Code + short report
**Estimated effort:** 4-6 hours
**Submit as:** `experiments/reports/assignment_04_<your_name>.md` + the quantized `.onnx` files + benchmark JSONs.

---

## Learning outcomes assessed

By submitting this assignment you demonstrate that you can:

1. Quantize a model to INT8 (dynamic and static) using `src.optimization`.
2. Measure the size / latency / accuracy trade-off **with real data** for calibration and evaluation.
3. Decide whether to ship a quantized model based on a defensible trade-off.

---

## Task

### Required

Pick **one** image classifier (your own, or a torchvision one — Chapters 5 / 8 use MobileNetV3-Small as the example).

Produce **three** ONNX files:

1. FP32 baseline (from Chapter 5 export).
2. INT8 dynamic.
3. INT8 static, **calibrated on at least 100 real images** from a held-out set.

### Required data

Use **real images** for calibration and evaluation — not random tensors. Acceptable sources:

- CIFAR-10 (32×32, resize to 224×224 for ImageNet classifiers).
- A small ImageNet subset (e.g. `imagenette` — 10 classes, ~10k images, easy to download).
- Your own set of ≥100 images from a classification task.

You do **not** need to retrain anything. Calibration uses inputs only; no labels are required for calibration. Labels are only needed if you want to compute top-1 accuracy (recommended).

### Required table

Produce a benchmark table for the three configurations on your target device (laptop CPU is fine):

| Config | Size (MB) | P50 ms | P95 ms | FPS estimate | argmax agreement vs FP32 | top-1 vs labels (if you have them) |
|---|---|---|---|---|---|---|
| FP32 | | | | | 1.000 (reference) | |
| INT8 dynamic | | | | | | |
| INT8 static (real calibration) | | | | | | |

### Required short analysis (~500 words)

Answer each:

1. **Size win:** how much smaller is each INT8 file vs FP32? Was it close to the theoretical 4×?
2. **Speed win or loss:** did INT8 actually speed up inference on your hardware? Why or why not?
3. **Accuracy:** if you have labels, what was the top-1 drop? If not, what was the argmax-agreement vs FP32?
4. **Which would you ship,** and why? (Cite the trade-off you accepted; don't say "all of them.")
5. **Where would the speed picture change?** Name one target hardware (Jetson INT8, Coral TPU, Intel NPU, RPi + TFLite ARM, etc.) where you would expect INT8 to win the speed comparison, and explain why.
6. **Honest limitations:** what about your calibration set might bias the result? (Coverage of classes, lighting, resolution, etc.)

---

## Grading rubric (100 points)

| Criterion | Points |
|---|---|
| Produced FP32, INT8 dynamic, INT8 static files | 15 |
| Calibration uses ≥100 REAL images (not random) | 15 |
| Table filled in with all required columns | 15 |
| Reported argmax agreement (and top-1 if labels available) | 10 |
| Analysis answers all 6 questions | 25 |
| Made a defensible "which to ship" decision | 10 |
| Honest disclosure of limitations | 5 |
| Files + JSON benchmarks saved to `experiments/` | 5 |
| **Total** | **100** |

---

## Common mistakes that lose points

- Calibrating with random tensors. The calibration set must be real images representative of inference inputs.
- Claiming an INT8 speedup without measuring it (or measuring on a different machine).
- Reporting only the size win, skipping latency and accuracy.
- Picking a model where quantization is known to break (e.g. very small custom models with unusual ops) and then complaining.
- "Argmax agreement = 100%" on 5 images is not a real claim. Use at least 100 evaluation images.

---

## Stretch goals (not graded)

- Repeat on a **larger model** (ResNet50). Does INT8 win the speed comparison more clearly there?
- Try **FP16** as well via TensorRT or OpenVINO (if you have the hardware).
- Plot accuracy-vs-latency as a scatter, with one point per config.
- Compare ORT INT8 latency to OpenVINO INT8 latency on the same machine — sometimes the runtime matters more than the precision.
