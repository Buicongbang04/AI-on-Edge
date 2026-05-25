# Lab 05 — Real-time camera AI

**Chapter:** 9 — Real-time camera AI with OpenCV
**Prerequisites:** finished `chapter_06_camera_inference_opencv.ipynb`.
**Estimated effort:** 2-3 hours.

The goal of this lab is to take the reference `src/inference/camera_loop.py` and (1) run it live on a webcam, (2) measure end-to-end FPS, (3) intentionally introduce a bottleneck and fix it, and (4) write a short report.

You can do the lab on a **laptop with a webcam**, a **Raspberry Pi 5 with a USB or CSI camera**, or a **Jetson** (any model). No two will give the same FPS — that is the point.

---

## Part 1 — Live camera demo (30 minutes)

1. Run the script live:

   ```bash
   python src/inference/camera_loop.py \
       --model experiments/exported_models/mobilenet_v3_small.onnx
   ```

2. Confirm: a window opens, your video appears, the top of the window shows a predicted class + FPS. Try pointing the camera at different objects to see the predictions change.

3. Read the on-screen FPS. **Write it down.** This is your baseline.

4. Press `q` (or ESC) to quit.

---

## Part 2 — Headless FPS benchmark (30 minutes)

You need a fixed-source baseline so the numbers are reproducible (the live webcam will fluctuate).

1. If you don't already have `datasets/sample_video.mp4`, generate it with the script from the chapter notebook (or use any short video you have).

2. Run headless on the file:

   ```bash
   python src/inference/camera_loop.py \
       --model experiments/exported_models/mobilenet_v3_small.onnx \
       --source datasets/sample_video.mp4 \
       --no-display --max-frames 200
   ```

3. Note the final FPS. This is your reproducible baseline.

4. (Optional, recommended) Wrap the loop with the `src.benchmark.benchmark_fps` API and record the result in `experiments/benchmark_results/`. That gives you P50 / P95 numbers, not just average FPS.

---

## Part 3 — Introduce a bottleneck on purpose (30 minutes)

The point: get a feel for *which stage* dominates by deliberately stressing one.

Pick **one** of:

- **Quality:** swap the model from MobileNetV3-Small to ResNet50 (export it with `src/export/export_onnx.py --model resnet50`). Re-run. How much did FPS drop?
- **Resolution:** change the script to pass through 448×448 instead of 224×224 (modify the preprocess) and re-run. How much did FPS drop?
- **Capture:** add `time.sleep(0.05)` inside the loop *after* `cap.read()` to simulate a slow camera. Where does FPS land?
- **Display:** remove `--no-display` and let `cv2.imshow` run. Does displaying the frames cost anything?

Record the new FPS for whichever you picked.

---

## Part 4 — Fix it (30 minutes)

Apply at least **one** optimization from Chapter 9, §6:

- `--skip 1` or `--skip 2` (run inference less often).
- Quantize the model (Ch 8) and rerun with the INT8 file.
- Downscale the camera before preprocessing.
- Threaded capture (write a small async producer).

Record the new FPS after the fix.

---

## Part 5 — Report (30 minutes)

Submit `experiments/reports/lab_05_<your_name>.md` with the following sections:

```markdown
# Lab 05 report

## Hardware tested
- Device: <laptop / Pi 5 / Jetson Orin Nano / ...>
- Camera: <built-in webcam / USB / CSI / ...>
- Runtime: ONNX Runtime CPU (or CUDA / OpenVINO / TensorRT)

## Baseline
- Live webcam FPS: ...
- Headless on sample_video.mp4 (200 frames): ...

## Induced bottleneck
- I changed: ...
- New FPS: ...
- The bottleneck stage was clearly: <capture / preprocess / inference / display>

## Fix applied
- I applied: ...
- New FPS: ...
- How close did it get back to baseline?

## Take-aways (≤200 words)
- Which stage is the natural bottleneck on my hardware, and why.
- What single optimization had the largest effect.
- What I would try next.
```

---

## Grading rubric (100 points)

| Criterion | Points |
|---|---|
| Ran the live demo successfully | 15 |
| Ran the headless baseline and recorded numbers | 15 |
| Introduced one bottleneck on purpose | 15 |
| Identified WHICH stage the bottleneck dominated (with numbers) | 15 |
| Applied at least one optimization and re-benchmarked | 20 |
| Report has all 5 sections, with concrete numbers | 15 |
| Take-aways are specific (not generic "I learned about pipelines") | 5 |
| **Total** | **100** |

---

## Common pitfalls

- Forgetting to put `--no-display` for headless runs; the script will hang waiting for X.
- Comparing webcam FPS to headless FPS — they are different experiments. Pick one for the baseline.
- Optimizing a stage that wasn't the bottleneck. Measure first.
- Confusing the camera's native FPS (e.g. 30) with the pipeline's FPS — the latter is capped by the former unless you do something clever.
