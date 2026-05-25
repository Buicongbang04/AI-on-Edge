# Deployment notes: runtime

**Template** — fill in one of these per project. The final-project rubric (Ch 20) requires this file to be present and complete.

---

## Identification

| Field | Value |
|---|---|
| Project name | <e.g. helmet-detector-v2> |
| Owner | <name / team> |
| Last updated | <YYYY-MM-DD> |

---

## Model artifact

| Field | Value |
|---|---|
| Architecture | <e.g. YOLOv8n> |
| Source weights | <git hash / dataset SHA of training run> |
| Exported format | <onnx / tflite / engine / xml+bin> |
| File path on device | <e.g. /opt/edge-ai/models/yolov8n_v1.0.3.onnx> |
| Size | <MB> |
| SHA-256 checksum | <as produced by `src.edgeops.compute_model_checksum`> |
| Precision | <FP32 / FP16 / INT8> |
| Input shape | <e.g. (1, 3, 640, 640)> |
| Preprocessing | <mean / std / resize / color> |

---

## Runtime

| Field | Value |
|---|---|
| Runtime | <e.g. ONNX Runtime 1.26.0> |
| Execution provider | <e.g. CUDAExecutionProvider, TensorrtExecutionProvider, OpenVINO NPU> |
| Threading | <intra_op / inter_op counts> |
| Session options | <graph_opt level, IO binding, etc.> |

---

## Hardware target

| Field | Value |
|---|---|
| Device | <e.g. NVIDIA Jetson Orin Nano Developer Kit 8GB> |
| OS | <e.g. Ubuntu 22.04, JetPack 6.2> |
| Power mode | <e.g. nvpmodel mode 0 (MAXN_SUPER)> |
| Storage | <SD / eMMC / NVMe> |
| Network | <Ethernet / Wi-Fi / cellular / none> |

---

## Performance

| Metric | Value |
|---|---|
| `model_only` P50 latency | <ms> |
| `model_only` P95 latency | <ms> |
| End-to-end P95 latency | <ms> |
| FPS end-to-end | <fps> |
| Peak CPU RSS | <MB> |
| Peak GPU VRAM | <MB or n/a> |
| Sustained temperature | <°C> |
| Power | <W> |

All measured with `src.benchmark.bench_full` and stored in `experiments/benchmark_results/`. Reference the JSON file path here.

---

## Configuration

| Param | Value | Why |
|---|---|---|
| confidence_threshold | 0.30 | Optimized for recall on the rare class |
| nms_iou | 0.45 | YOLO default |
| input_resolution | 640 | Balance speed and small-object recall |
| skip_frames | 1 | Process every other frame; sufficient for 30 FPS camera |
| <other> | | |

---

## Deployment process

1. Build / export step: `<script + commit hash>`
2. Validate ONNX: `python src/export/export_onnx.py --no-validate=false`
3. Copy artifact to device: `<scp / rsync / OTA tool>`
4. Verify checksum on device: `python -c "from src.edgeops import compute_model_checksum; print(compute_model_checksum('...'))"`
5. Restart inference service: `<systemd / docker / pm2>`
6. Health check: `python <health_check_script>` — expects class X with confidence ≥ Y on `assets/canary.jpg`.

---

## Rollback plan

- Previous version on device: `<path>`
- Switching mechanism: <symlink / config flag>
- Verification after rollback: re-run health check.
- Estimated rollback time: <minutes>.

---

## Update channel

- Canary device(s): <list>
- 24h soak: yes / no
- Rollout stages: canary → 10% → 100%
- Auto-rollback trigger: <e.g. fallback rate > 20% over 5 min>
