# Project 01 — Camera image classifier (starter)

A starting point for your first end-to-end edge AI project: a real-time image classifier on a webcam (or video file), using the reference scripts from Chapters 3, 6, and 9.

This folder is intentionally minimal — there is no new code here. Everything you need already lives in `src/` and the chapter notebooks. Use this folder to fork into your own project (rename it) and add your model, your config, and your benchmark results.

---

## What this project demonstrates (in scope)

- Loading a pretrained or fine-tuned PyTorch classifier (Chapter 3).
- Exporting it to ONNX (Chapter 5).
- Running it via ONNX Runtime in a live camera loop (Chapter 6 + 9).
- Benchmarking the end-to-end pipeline (Chapter 4).

---

## Quick start (uses the repo's reference scripts)

```bash
# 1. Inference on a single image
python ../../src/inference/infer_pytorch.py --image ../../datasets/sample.jpg

# 2. Export to ONNX (if you haven't yet)
python ../../src/export/export_onnx.py --model mobilenet_v3_small

# 3. Live camera loop (press q to quit)
python ../../src/inference/camera_loop.py \
    --model ../../experiments/exported_models/mobilenet_v3_small.onnx

# 4. Benchmark
python -m src.benchmark --model mobilenet_v3_small --device cpu --pipeline
```

---

## How to make this *your* project

1. Fork this folder to `projects/<my_classifier>/`.
2. Copy `src/inference/camera_loop.py` into it and edit for your model / classes.
3. Add a `config.yaml` for thresholds and paths.
4. Add a `results/` directory with your benchmark JSON and a demo video.
5. Fill in `deployment_notes/runtime.md` and `deployment_notes/safety.md` (templates in repo root `deployment_notes/`).
6. Write a project README that explains your problem, hardware, and results.

For inspiration on a more developed project, see `projects/project_02_yolo_realtime_camera/`.
