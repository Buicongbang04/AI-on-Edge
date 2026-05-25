# Project 02 — YOLO real-time object detection on the edge

A complete mini-project that takes you from a pretrained YOLOv8 detector to a real-time camera demo, an ONNX export, and a benchmark report. It is the worked example for Chapter 10.

---

## Goal

Build a real-time object detector that:

- Reads frames from a webcam (or a video file).
- Runs YOLOv8 inference on each frame.
- Overlays detected boxes, classes, and confidence.
- Reports an end-to-end FPS counter.
- Exports the model to ONNX for redeployment.

---

## Hardware tiers

| Tier | Hardware | Expected FPS (640×640, YOLOv8n) |
|---|---|---|
| Level 1 | Laptop CPU | 5-15 FPS |
| Level 1 | Laptop with NVIDIA GPU | 30-60 FPS |
| Level 2 | Raspberry Pi 5 + Coral / Hailo | 15-30 FPS |
| Level 3 | NVIDIA Jetson Orin Nano + TensorRT FP16 | 60-90 FPS |

For the first run, use a laptop and accept that CPU FPS will be modest.

---

## Files in this project

```
project_02_yolo_realtime_camera/
├── README.md              — this file
├── config.yaml            — runtime configuration
├── run_image.py           — single-image detection demo
├── run_video.py           — video / camera detection demo
├── export_onnx.py         — export YOLOv8 to ONNX (one-line wrapper)
└── results/               — output frames, exported model, benchmark JSONs
```

---

## Quick start

```bash
cd projects/project_02_yolo_realtime_camera

# 1. Run on a single image (YOLOv8n is auto-downloaded the first time)
python run_image.py --image ../../datasets/sample.jpg --model yolov8n.pt

# 2. Run on the synthetic video file
python run_video.py --source ../../datasets/sample_video.mp4 \
    --model yolov8n.pt --no-display --max-frames 60 \
    --record results/sample_video_annotated.mp4

# 3. Run live on a webcam
python run_video.py --source 0 --model yolov8n.pt

# 4. Export to ONNX so it can run via ONNX Runtime / TensorRT
python export_onnx.py --model yolov8n.pt --imgsz 640
# -> writes yolov8n.onnx
```

---

## Configuration

`config.yaml` keeps the per-deployment settings in one place so you don't repeat flags. The runner scripts read it if `--config config.yaml` is passed.

```yaml
model: yolov8n.pt
imgsz: 640
conf: 0.25
iou: 0.45
device: auto    # 'auto', 'cpu', 'cuda', 0 (GPU id)
classes: null   # or a list of class indices to filter (COCO ids)
```

For helmet / PPE detection, replace `model:` with the path to your fine-tuned weights and `classes:` with the relevant indices.

---

## What the scripts do

### `run_image.py`

- Loads the model (`.pt` or `.onnx` — ultralytics handles both).
- Runs detection on a single image at the configured confidence / IoU.
- Saves the annotated image to `results/`.
- Prints latency.

### `run_video.py`

- Loops over a video file or webcam.
- Annotates each frame with `result.plot()` (ultralytics' overlay) plus an FPS counter.
- Optional `--record path` saves the annotated video.
- Optional `--no-display` for headless runs.

### `export_onnx.py`

- Wraps `model.export(format="onnx", imgsz=...)` from ultralytics.
- Writes a `.onnx` file alongside the weights.
- The resulting ONNX can be loaded by `src/inference/infer_tensorrt.py` (after `trtexec`) or by OpenVINO / ONNX Runtime.

---

## Use-case ideas

The same pipeline solves many real problems by swapping the model:

- **People counting** — filter to class 0 (`person`), aggregate per minute.
- **Safety helmet detection** — fine-tune YOLOv8 on a helmet dataset; filter to the helmet/no-helmet classes.
- **Product defect detection** — fine-tune on your defect dataset.
- **Vehicle / license plate** — chain YOLO + an OCR step.

Each case adds postprocessing and an alerting layer on top of the same loop.

---

## Reporting

Drop your benchmark numbers into `results/benchmark.json` (use the `src.benchmark.benchmark_fps` API from Chapter 4). The final-project rubric (Ch 20) reads from here.

A typical report includes:

| Field | Example |
|---|---|
| Hardware | "Jetson Orin Nano 8 GB, 25 W MAXN_SUPER, fan-cooled" |
| Runtime | "ONNX Runtime CUDAExecutionProvider" |
| Precision | "FP16 (TensorRT engine)" |
| Input | "640×640 single image" |
| End-to-end FPS | 72 |
| P95 latency | 18 ms |
| Notes | "First-build engine took 92 s; cached afterwards." |
