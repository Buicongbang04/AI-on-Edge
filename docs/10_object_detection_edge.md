# Chapter 10 — Object detection on the edge

> **Goal:** Move from image classification (a single label per frame) to **object detection** (multiple boxes + classes + scores per frame). By the end of this chapter you should be able to run YOLOv8 on an image, video, or webcam, tune the confidence and IoU thresholds, export to ONNX, and benchmark FPS.

Object detection is the workhorse of real-world camera AI: people counting, vehicle / license-plate recognition, PPE compliance, retail analytics, drone surveillance, manufacturing defect localization. The course adopts YOLOv8 (Ultralytics) as the canonical detector because it has the cleanest training and export tooling in 2026.

---

## 1. What object detection produces

A classifier returns one label per image. A detector returns a **list** of:

```
(class_id, x_min, y_min, x_max, y_max, confidence)
```

per image, often dozens or hundreds. The pipeline then:

1. Filters by confidence (`conf >= 0.25` is the YOLO default).
2. Applies **NMS** (non-maximum suppression, IoU threshold ~0.45) to remove duplicate boxes covering the same object.
3. Optionally filters by class (`only show "person"`).
4. Hands the surviving boxes to your application logic.

NMS is the silent half of object detection. Without it you get many overlapping boxes per object; with it you get one. The course relies on the runtime to apply NMS — ultralytics, ONNX Runtime via the YOLO-fused export, and TensorRT all do this for you.

---

## 2. Why YOLO

YOLO (You Only Look Once) family detectors run the whole image through one CNN that produces all boxes at once — there is no separate region-proposal step. This makes YOLO:

- Fast (especially the `n`, `s` sizes).
- Easy to deploy (single-stage architecture; ONNX-friendly).
- Well-supported with quantization and edge runtimes.

Variants used in this course:

| Model | Params | Size FP32 | mAP COCO (typical) | Use |
|---|---|---|---|---|
| YOLOv8n | 3.2 M | ~12 MB | ~37 | Pi 5, Jetson, anywhere — first try |
| YOLOv8s | 11.2 M | ~43 MB | ~45 | Better accuracy, still fast on Jetson |
| YOLOv8m | 25.9 M | ~99 MB | ~50 | Larger Jetson / NUC; multi-stream |
| YOLOv8l, x | 43+ M | 165+ MB | 52+ | Cloud or beefy edge; usually not edge |

YOLOv8n is the course default. Start there, escalate only if recall is too low.

---

## 3. The Ultralytics API

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")           # auto-downloads on first use
results = model.predict(             # accepts path / numpy frame / batch
    source="image.jpg",
    imgsz=640,                        # input resolution (multiple of 32)
    conf=0.25, iou=0.45,
    device=None,                     # None = auto (CUDA if available, else CPU)
    verbose=False,
)
r = results[0]
print(r.boxes)                       # tensor: (N, 6) = [x1, y1, x2, y2, conf, cls]
annotated = r.plot()                 # numpy BGR image with boxes drawn
```

The same `model.predict(...)` accepts:

- A single image path or numpy frame.
- A list of images (batched).
- A video path or stream URL (auto-iterates over frames).
- A webcam id like `"0"`.

For real-time loops the course uses `cv2.VideoCapture` directly so we control FPS counters and recording — see `projects/project_02_yolo_realtime_camera/run_video.py`.

---

## 4. Confidence and IoU thresholds

Two knobs control the precision / recall trade-off:

- **`conf`** (default 0.25) — minimum confidence to keep a detection.
- **`iou`** (default 0.45) — NMS IoU threshold (boxes with IoU above this are de-duplicated).

For *safety-critical* use cases (helmet detection, PPE) you want **high recall**: lower `conf` (e.g. 0.15) and accept more false positives. For *cosmetic* uses (people counting in a retail store), higher `conf` (e.g. 0.4) reduces clutter.

Sweep these on your validation set and pick the operating point that meets your business-cost ratio (Ch 1, §4.3).

---

## 5. Input resolution vs FPS

YOLO accepts input resolutions in multiples of 32 (`imgsz=320, 416, 640, 1280, ...`). Compute scales roughly as `H × W`.

| imgsz | Speed | Detail |
|---|---|---|
| 320 | ~4× faster than 640 | Misses small objects (badges, distant pedestrians) |
| 416 | ~2× faster than 640 | A good middle ground for Jetson Nano |
| 640 | default | Good default for most use cases |
| 1280 | ~4× slower than 640 | Needed for small-object detection (drones, satellite, fine defects) |

Always benchmark *your* use case at the resolution you plan to ship.

---

## 6. Lightweight YOLO alternatives

If YOLOv8n is still too heavy on your edge device:

- **YOLOv8n + INT8 quantization** (Ch 8) — often 2-3× speedup on NPU / TPU / TFLite.
- **YOLO-NAS** (Deci) — sometimes faster and more accurate than YOLOv8n at the same size.
- **NanoDet-Plus, PicoDet** — sub-2M-parameter detectors for very constrained devices.
- **MobileNet-SSD** — older but very small and well-supported by TFLite.

For final projects, the course recommends sticking with YOLOv8 unless your latency budget forces an alternative.

---

## 7. Basic tracking

If you need to count individual objects (e.g. "how many *unique* people walked by"), you also need a **tracker** that assigns persistent IDs across frames. Ultralytics ships with two:

```python
results = model.track(source="video.mp4", persist=True, tracker="bytetrack.yaml")
# r.boxes.id  → tracker ID per detection
```

- `bytetrack.yaml` — fast, robust default.
- `botsort.yaml` — slower but better at re-identifying after occlusion.

Tracking is out of scope for the chapter's grading rubric but appears in the final project for any "counting" or "behavior" use case.

---

## 8. The project: `projects/project_02_yolo_realtime_camera/`

This is the worked example for Chapter 10. It has:

- `run_image.py` — single-image YOLO detection with annotated output.
- `run_video.py` — real-time camera / video loop with FPS counter.
- `export_onnx.py` — export to ONNX for redeployment.
- `config.yaml` — model / imgsz / thresholds.

See `projects/project_02_yolo_realtime_camera/README.md` for the full quick-start and the hardware-tier table.

---

## 9. Use cases

Each of the following is achievable by changing the YOLO model + class filter:

| Use case | YOLO weight | Class filter | Postprocess |
|---|---|---|---|
| People counting | yolov8n.pt | `[0]` (person) | Unique track IDs over time |
| Safety helmet (PPE) | fine-tuned on a helmet dataset | helmet, no-helmet | Alert on no-helmet for >K frames |
| Product detection | fine-tuned on factory dataset | product, defect | Per-product flow |
| Vehicle / license plate | yolov8n.pt + LPR step | car, motorcycle, truck, bus | Chain with ANPR |
| Wildlife camera | fine-tuned on local fauna | per-species | Log + telemetry |

For the final project (Ch 20), pick one use case and demonstrate it end-to-end with benchmark + deployment notes + safety notes.

---

## 10. Common pitfalls

- **Forgetting to set `conf`** — get hundreds of low-confidence boxes from a busy frame. Raise `conf`.
- **Comparing models at different `imgsz`** — speed and mAP differ a lot. Pin resolution when benchmarking.
- **Treating per-image latency as the FPS** — YOLO does some postprocessing inside `predict()`. The end-to-end FPS is lower; measure with the camera-loop pattern, not single-image latency.
- **Quantization breaks YOLO output decoding** if your runtime does not include the YOLO postprocess. `ultralytics.export()` bakes the decoder into the ONNX; if you build a custom export, replicate it.
- **TensorRT INT8 of YOLO** needs calibration over real frames, not random tensors (Ch 8).

---

## 11. What you should be able to do after this chapter

- Run YOLOv8 on an image, video, and webcam.
- Tune `conf` and `iou` for your precision / recall target.
- Export YOLOv8 to ONNX and confirm it runs.
- Benchmark FPS end-to-end.
- Decide which YOLO variant + resolution fits your edge device.

---

## 12. Files produced by this chapter

- `docs/10_object_detection_edge.md` — this file.
- `notebooks/chapter_07_yolo_edge_detection.ipynb` — chapter notebook.
- `projects/project_02_yolo_realtime_camera/` — full mini-project (README, runners, config).
