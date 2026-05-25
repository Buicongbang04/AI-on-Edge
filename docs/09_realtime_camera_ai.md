# Chapter 9 — Real-time camera AI with OpenCV

> **Goal:** Build a real-time inference loop on a webcam (or video file): capture → preprocess → infer → postprocess → overlay → display. By the end of this chapter you should be able to write a camera loop with an FPS counter, optimize it with resize / frame-skip / threading, and benchmark its end-to-end performance.

This is the first chapter where the model leaves the notebook and runs against a live data source. Everything you built in Chapters 3-8 — PyTorch / ONNX / quantization / benchmarking — comes together as a streaming pipeline.

---

## 1. The pipeline

Every real-time camera AI application has the same shape:

```
[ camera ] → capture → preprocess → infer → postprocess → overlay → [ display / log / MQTT ]
                |                                                          |
                └───────────── all of this runs every frame ───────────────┘
```

Each stage adds latency. The end-to-end FPS is bounded by the *slowest* stage. The work is:

1. **Capture** — read a frame from `cv2.VideoCapture`.
2. **Preprocess** — convert color space, resize, normalize, batch-add.
3. **Infer** — run the model (PyTorch / ONNX Runtime / TensorRT / OpenVINO / TFLite).
4. **Postprocess** — softmax / argmax / NMS / format the output.
5. **Overlay** — draw label, FPS, boxes onto the frame.
6. **Display / output** — `cv2.imshow`, write to video, publish MQTT, log alert.

---

## 2. The minimal camera loop

```python
import cv2, time
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("mobilenet.onnx", providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name

cap = cv2.VideoCapture(0)        # 0 = default webcam
while True:
    ok, frame = cap.read()
    if not ok:
        break
    x = preprocess(frame)         # your function
    logits = sess.run(None, {in_name: x})[0]
    label  = postprocess(logits)  # argmax + class name
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
```

That is the whole architecture. Everything in this chapter is a variation on it.

---

## 3. The reference script

`src/inference/camera_loop.py` is the production-grade version. It adds:

- **FPS counter** computed from a sliding window of recent frame intervals.
- **Frame skipping** (`--skip N` to run inference every N+1 frames, keep last prediction otherwise).
- **Headless mode** (`--no-display`, for SSH / no-X environments).
- **Recording** (`--record out.mp4`, to capture the demo).
- **Frame budget** (`--max-frames`, for benchmark runs).

Example:

```bash
# webcam + display
python src/inference/camera_loop.py \
    --model experiments/exported_models/mobilenet_v3_small.onnx

# headless on a video file (CI / benchmark)
python src/inference/camera_loop.py \
    --model experiments/exported_models/mobilenet_v3_small.onnx \
    --source datasets/sample_video.mp4 \
    --no-display --max-frames 100 \
    --record experiments/reports/demo.mp4
```

On a 16-core x86 laptop with the MobileNetV3-Small ONNX, the pipeline runs **>100 FPS end-to-end** (well above any USB webcam's native 25-30 FPS, so the camera, not the model, is the bottleneck).

---

## 4. Preprocessing in OpenCV (not PIL)

The reference scripts in Chapters 3 / 6 use `torchvision` transforms (which expect PIL images). On a live camera loop you do not want to allocate a PIL Image per frame; the hot path should be all NumPy / OpenCV:

```python
def preprocess_frame(bgr_frame):
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    # resize so shorter side = 232, center-crop 224
    h, w = rgb.shape[:2]
    target = 232
    if h < w:
        new_h, new_w = target, int(w * target / h)
    else:
        new_h, new_w = int(h * target / w), target
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    y0 = (new_h - 224) // 2
    x0 = (new_w - 224) // 2
    crop = resized[y0:y0+224, x0:x0+224]
    chw = crop.astype(np.float32) / 255.0
    chw = chw.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return ((chw - mean) / std)[None, ...]
```

This is what the reference script uses. The math matches `MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()`.

**The preprocessing must match training.** If you trained with a different resize / normalization, replicate it. Mismatch is the #1 silent bug.

---

## 5. The FPS counter

A naive FPS counter divides total frames by total elapsed time. That is fine for averages but misleading for monitoring:

```python
# WRONG — long-running average, hides transients
avg_fps = frame_count / (time.perf_counter() - start_time)
```

The reference script uses a **sliding window**:

```python
recent = []
last = time.perf_counter()
while True:
    # ... do work ...
    now = time.perf_counter()
    recent.append(now - last)
    last = now
    if len(recent) > WINDOW:
        recent.pop(0)
    fps = len(recent) / sum(recent)
```

This reports the *current* throughput, not the historical average — exactly what an operator wants to see.

---

## 6. Optimization tricks

When end-to-end FPS is below your budget, in priority order:

1. **Run inference every Nth frame** (`--skip N`). For most camera demos, 15 FPS predictions on a 30 FPS feed look the same; capture is the only thing the user notices.
2. **Drop the input resolution** to 160×160 or 192×192 if the model permits it. Inference time scales with H × W.
3. **Quantize the model** (Ch 8). On NPU / TPU / Jetson this is a 2-3× win.
4. **Pick a smaller architecture.** MobileNetV3-Small > ResNet50 on edge.
5. **Move to a faster runtime** for your hardware: ORT → OpenVINO / TensorRT (Ch 7).
6. **Async capture** — separate thread for `cap.read()` so capture doesn't block inference.
7. **MIPI CSI camera** instead of USB on Jetson / Pi (much lower capture latency).

You usually do not need all six. Apply them in order and re-benchmark after each.

---

## 7. Threading and async capture

USB webcam capture is *blocking*: `cap.read()` waits for the next frame from the camera. If your inference is faster than the camera, you waste cycles waiting. If your inference is slower, frames buffer up and the system lags.

A simple async-capture pattern:

```python
import threading, queue

frame_q = queue.Queue(maxsize=2)   # only keep the latest few frames

def producer(cap, q):
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if q.full():
            q.get_nowait()        # drop oldest, keep latest
        q.put(frame)

t = threading.Thread(target=producer, args=(cap, frame_q), daemon=True)
t.start()

while True:
    frame = frame_q.get()
    # ... inference ...
```

This decouples capture from inference. Use `maxsize=1-2` so you always work on the *latest* frame; dropping old frames is correct for real-time displays.

---

## 8. Common pitfalls

- **`cap.read()` returns the same frame repeatedly.** The camera disconnected or the codec hung. Treat `False` from `cap.read()` as failure and reopen.
- **First few frames are black or have wrong exposure.** USB webcams have a startup ramp; throw away the first ~20 frames.
- **Latency higher than expected** even though inference is fast. The bottleneck is usually capture decode (especially MJPEG) or display. Test with `cv2.CAP_PROP_FPS`.
- **`cv2.imshow` blocks on some Wayland systems.** On Linux desktop, prefer X11 for development; in production deployments use `--no-display`.
- **Autoexposure jitter** dominates the latency variance. Lock exposure via `cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, …)` (camera-dependent) when benchmarking.
- **`destroyAllWindows()` not called.** OpenCV windows can stick around after a crash; always wrap the loop in `try / finally`.

---

## 9. What you should be able to do after this chapter

- Build a real-time camera loop that captures, preprocesses, runs inference, overlays results, and displays them.
- Implement an FPS counter with a sliding window.
- Optimize a slow pipeline with resize, frame-skipping, quantization, or async capture.
- Benchmark the pipeline with `src.benchmark.benchmark_fps` (Chapter 4).
- Record an annotated demo video.

---

## 10. Files produced by this chapter

- `docs/09_realtime_camera_ai.md` — this file.
- `src/inference/camera_loop.py` — reference real-time camera classifier.
- `notebooks/chapter_06_camera_inference_opencv.ipynb` — chapter notebook (frame-level walkthrough).
- `labs/lab_05_realtime_camera.md` — guided lab.
- `projects/project_01_camera_classifier/` — full mini-project that builds on this script.
