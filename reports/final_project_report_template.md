# <Project name> — final project report

**Course:** Edge AI and Introductory Physical AI
**Author:** <your name>
**Date:** <YYYY-MM-DD>
**Repo:** <link to your fork>

---

## 1. Problem statement

*What does this system do, for whom, in what setting, with what business or operational consequence?*

<paragraph>

Why does this need to be on the edge rather than the cloud? Cite at least one of: latency, bandwidth, privacy, offline operation, physical control.

<paragraph>

---

## 2. Device / simulator environment

| Field | Value |
|---|---|
| Device | <e.g. NVIDIA Jetson Orin Nano 8GB> |
| OS | <e.g. Ubuntu 22.04 + JetPack 6.2> |
| RAM available to process | <e.g. 6 GB after system> |
| Storage | <e.g. 64 GB NVMe> |
| Camera / sensor | <e.g. CSI IMX477, RTSP 1080p, IMU @ 1 kHz> |
| Python | <3.11.x> |
| Key library versions | <torch, onnxruntime, opencv, ultralytics, ...> |

---

## 3. Dataset

| Field | Value |
|---|---|
| Source | <e.g. MVTec AD bottle subset / custom / Speech Commands> |
| Size | <train / val / test counts> |
| Splits | <random / by time / by lot> |
| Labels | <how labeling rule was set> |
| Notes | <imbalance, lighting variations, etc.> |

---

## 4. Model baseline

| Field | Value |
|---|---|
| Architecture | <e.g. MobileNetV3-Small, YOLOv8n, autoencoder> |
| Parameters | <millions> |
| Training | <epochs, batch size, optimizer, LR; or "pretrained, no training"> |
| FP32 quality metric (held-out set) | <e.g. top-1 acc 0.92, mAP@0.5 = 0.78, recall on defect 0.94> |
| FP32 latency (model-only) | <e.g. mean 5.0 ms, P95 5.5 ms> |

---

## 5. Export / optimization

- Export script: `<path/to/export.py>`
- Format: `<onnx / tflite / engine>`
- Opset / target: `<e.g. opset 17, TensorRT 10.x>`
- Validation: max abs diff vs PyTorch = <…>; argmax agreement = <…>; on N held-out samples.
- Optimization applied: `<none / FP16 / INT8 dynamic / INT8 static / pruning / distillation>`
- Calibration data (if INT8 static): <description, count>

Result table:

| Variant | Size (MB) | P50 ms | P95 ms | Accuracy vs FP32 |
|---|---|---|---|---|
| FP32 ONNX | | | | 1.000 (reference) |
| FP16 / INT8 | | | | |
| (additional) | | | | |

---

## 6. Runtime

| Field | Value |
|---|---|
| Runtime | <e.g. ONNX Runtime 1.26.0> |
| Execution provider | <e.g. CUDAExecutionProvider, TensorrtExecutionProvider, OpenVINO NPU> |
| Threading | <intra_op = …, inter_op = …> |
| Session options | <graph_opt level, IO binding, profiling> |

---

## 7. Benchmark

JSONs: `experiments/benchmark_results/<...>.json` (list them).

| Metric | Value |
|---|---|
| `model_only` mean | <ms> |
| `model_only` P50 | <ms> |
| `model_only` P95 | <ms> |
| `model_only` P99 | <ms> |
| End-to-end mean | <ms> |
| End-to-end P95 | <ms> |
| FPS end-to-end | <fps> |
| Peak CPU RSS | <MB> |
| Peak GPU VRAM | <MB / n/a> |
| Sustained temperature | <°C / not measured> |
| Power | <W / not measured> |

---

## 8. Metric

- Quality metric + target: <e.g. recall on defect ≥ 0.95 at precision ≥ 0.80>
- Performance metric + target: <e.g. P95 ≤ 50 ms, FPS ≥ 20>
- Cost trade-off: <FN cost = $X, FP cost = $Y; which dominates>
- Achieved: <numbers>
- Decision (ship / iterate): <one sentence>

---

## 9. Demo

How to reproduce the demo from a fresh checkout:

```bash
# 1. Install
pip install -r requirements.txt

# 2. (optional) Download the trained weights
<wget / git lfs / hf download>

# 3. Export
python <path>/export.py

# 4. Run demo
python <path>/infer.py --source <input> --no-display --record results/demo.mp4

# 5. Benchmark
python <path>/benchmark.py
```

Demo artifact: `results/demo.mp4` or `results/screenshots/*.png`.

---

## 10. Error analysis

- Confusion matrix or per-class precision/recall: <attach plot>
- Where the model fails: <classes, lighting, motion, edge cases>
- Failure mode catalog: <e.g. "low light", "occlusion", "very small object">
- What changed when you swept the operating point (confidence threshold): <plot or table>
- Honest limits: <what is out of distribution; what you would not deploy yet>

---

## 11. Deployment notes

Link or copy: `deployment_notes/runtime.md` (see template in repo root).

Summary:

- Hardware target: <…>
- Update mechanism: <…>
- Rollback plan: <…>
- Health check: <…>

---

## 12. Risk / safety notes

Link or copy: `deployment_notes/safety.md` (see template in repo root).

Summary:

- Top 5 failure modes and responses: <…>
- Fallback strategy: <…>
- Operator override mechanism: <…>
- Out-of-scope (security / threats): <…>

---

## Self-check against rubric (100 points)

| Criterion | Self-score |
|---|---|
| Problem and edge constraints (10) | |
| Dataset and preprocessing (10) | |
| Model baseline (10) | |
| Export / optimization / runtime (15) | |
| Benchmark end-to-end (15) | |
| Metric appropriate to task (10) | |
| Demo runs (10) | |
| Error analysis (10) | |
| Deployment + safety notes (5) | |
| Repo clean / reproducible (5) | |
| **Total** | |
