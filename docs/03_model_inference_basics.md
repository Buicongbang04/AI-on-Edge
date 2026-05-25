# Chapter 3 — Inference basics: from checkpoint to prediction

> **Goal:** Re-train the deployment mindset: load a checkpoint, run inference cleanly, and produce honest per-image latency numbers. This is the smallest end-to-end deployment loop the course will build on top of for the next several chapters.

This chapter assumes you already know how to train a PyTorch model. What it teaches is the discipline of running a *trained* model: the small set of habits that separate a notebook that "works" from inference code you can ship.

---

## 1. Training mode vs inference mode

PyTorch modules have two operating modes:

- **Training mode** (`model.train()`): dropout drops, BatchNorm tracks running statistics, gradients accumulate, autograd builds a computation graph.
- **Inference mode** (`model.eval()` + `torch.no_grad()`): dropout passes through, BatchNorm uses stored statistics, gradients are not tracked, no graph is built.

This sounds obvious, but the two failures it causes are the most common bugs in deployed PyTorch code:

```python
# WRONG — model is still in training mode; BN/dropout will be active
model = torch.load("checkpoint.pt")
y = model(x)

# WRONG — grads are tracked; uses 2-3× more memory and runs slower
model.eval()
y = model(x)

# RIGHT — eval + no_grad
model.eval()
with torch.no_grad():
    y = model(x)
```

The reference inference script (`src/inference/infer_pytorch.py`) does both. Make this a reflex.

---

## 2. Loading a checkpoint

Three legal patterns you will see in the wild:

```python
# Pattern A — full module pickled (NOT recommended; tied to your code's class paths)
model = torch.load("model.pt", map_location="cpu")

# Pattern B — state_dict + architecture (PREFERRED)
model = MyModelClass(num_classes=10)
state_dict = torch.load("weights.pt", map_location="cpu")
model.load_state_dict(state_dict)

# Pattern C — pretrained from torchvision (what this course uses for Ch 3-4)
from torchvision import models
weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = models.mobilenet_v3_small(weights=weights)
```

For deployment, **always go through Pattern B or C**. Pattern A breaks if your class moves or your codebase reorganizes; that breakage shows up in production, not in tests.

Always pass `map_location="cpu"` when loading on a machine that may not have a GPU; then `.to(device)` after.

---

## 3. Preprocessing consistency

The single most common silent bug in deployed CV models is a mismatch between training preprocessing and inference preprocessing. If you trained with `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` (ImageNet stats) and you forget the normalization at inference, your model still produces *plausible-looking* outputs — just wrong.

The deployment rule is: **always re-use the exact preprocessing chain from training, ideally bundled with the model.**

`torchvision` makes this clean by attaching the right transform to each pretrained weight:

```python
weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
preprocess = weights.transforms()   # the official preprocessing for this checkpoint
x = preprocess(pil_image).unsqueeze(0)
```

For your own models, save the preprocessing config alongside the weights (image size, normalization mean/std, color space, BGR vs RGB) and load both together. The Chapter 19 EdgeOps log schema includes a `preprocessing_config` field for this reason.

---

## 4. Single image vs batch inference

For real-time edge inference you typically run **batch size = 1**: one frame, one prediction, as soon as possible. For offline scoring or throughput benchmarks, you run **larger batches** to amortize fixed costs.

```python
# Single image — for real-time camera or sensor loop
x = preprocess(img).unsqueeze(0)        # shape: (1, C, H, W)
y = model(x)

# Batch — for offline scoring
batch = torch.stack([preprocess(im) for im in images])  # shape: (B, C, H, W)
y = model(batch)
```

A common mistake: benchmarking on batch=32 and then deploying on batch=1. The per-image latency is *not* `batch_latency / batch_size` — fixed kernel-launch costs dominate at batch=1. Always benchmark in the configuration you ship.

---

## 5. Warm-up

The first few inferences after a fresh model load are slower because:

- PyTorch / CUDA may JIT-compile kernels.
- cuDNN searches for the best convolution algorithm for your input shape (the "cuDNN benchmark").
- Memory allocators warm up.
- Even on CPU, oneDNN and MKL have first-call overheads.

If you time the very first call, you measure the *warm-up* cost, not the steady-state cost. The fix is to **always warm up before timing**:

```python
# 3-10 warm-up iterations is usually enough on CPU; 10-20 on CUDA
for _ in range(10):
    _ = model(x)

# now time
times = []
for _ in range(100):
    start = time.perf_counter()
    _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times.append(time.perf_counter() - start)
```

The `torch.cuda.synchronize()` is critical on GPU: without it, `time.perf_counter()` returns before the kernel has actually finished, and you measure launch latency, not compute latency.

---

## 6. The right way to time

```python
import time

# CORRECT for both CPU and CUDA, assuming sync where applicable
start = time.perf_counter()
with torch.no_grad():
    y = model(x)
if device.type == "cuda":
    torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - start) * 1000
```

Do **not** use:

- `time.time()` (low-resolution, can jump backwards on NTP sync).
- `datetime.now()` (same problem, even slower).
- `%%timeit` magic without `torch.cuda.synchronize()` (under-reports GPU time).
- Wall-clock differences across a multi-second loop (jitter dominates).

`time.perf_counter()` is the right primitive. Chapter 4 builds the full benchmark template on top of it.

---

## 7. CPU vs GPU: when to use which

| Situation | Use |
|---|---|
| Model fits in RAM, batch=1, small architecture (MobileNet, EfficientNet-Lite) | **CPU** is often within a few × of GPU on a laptop; simpler |
| Larger architecture (ResNet-50, YOLO at 640×640, ViT) | **GPU** wins clearly |
| Real edge target is a Pi or NUC | **CPU** is the realistic prototype; benchmark there |
| Real edge target is a Jetson | **GPU** is the realistic prototype; use TensorRT, not raw PyTorch (Ch 7) |
| You want the most portable code | **CPU** path always works; GPU path requires CUDA |

In this course we use CPU as the default in Chapters 3-4 (so everyone can run the code). Chapters 5-8 introduce ONNX Runtime and quantization, which together close most of the CPU-vs-GPU gap on small models.

---

## 8. The reference script

`src/inference/infer_pytorch.py` is the reference Python script for this chapter. It supports:

```bash
# Default: MobileNetV3-Small on CPU, the sample image
python src/inference/infer_pytorch.py --image datasets/sample.jpg

# A folder of images
python src/inference/infer_pytorch.py --image-dir datasets/test_images

# Use CUDA if available, otherwise fall back to CPU
python src/inference/infer_pytorch.py --image datasets/sample.jpg --device auto

# Swap to a heavier model
python src/inference/infer_pytorch.py --image datasets/sample.jpg --model resnet50

# Reduce warm-up if you're just sanity-checking
python src/inference/infer_pytorch.py --image datasets/sample.jpg --warmup 1
```

Read the script. It is short and is the same pattern every later chapter follows.

---

## 9. The chapter notebook

`notebooks/chapter_01_latency_benchmarking.ipynb` walks through:

1. Loading the model (Pattern C above) and verifying mode/grad state.
2. Showing the cost of *not* using `eval()` and `no_grad()`.
3. Cold vs warm latency: how the first few calls differ from steady state.
4. CPU vs GPU comparison if a GPU is available.
5. Single-image vs small-batch comparison.
6. A histogram of measured latencies and a P50 / P95 print-out.

This is the smallest honest benchmark you can run; Chapter 4 generalizes it.

---

## 10. What you should be able to do after this chapter

- Load a PyTorch model in inference mode (`eval` + `no_grad`).
- Apply the right preprocessing for the model you loaded.
- Run inference on a single image and a folder of images.
- Time inference correctly, including warm-up and CUDA sync.
- Read a top-k prediction list and understand which class scored what probability.

---

## 11. Files produced by this chapter

- `docs/03_model_inference_basics.md` — this file.
- `src/inference/infer_pytorch.py` — the reference inference script.
- `notebooks/chapter_01_latency_benchmarking.ipynb` — the chapter notebook.
- `datasets/sample.jpg` — small synthetic image for smoke-testing.
