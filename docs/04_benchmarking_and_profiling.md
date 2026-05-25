# Chapter 4 — Benchmarking: Latency, FPS, Memory, Throughput

> **Goal:** Measure inference performance *honestly*. By the end of this chapter you should be able to write a benchmark that reports model-only latency percentiles, end-to-end FPS, peak memory, and the metadata needed to compare runs across machines — and you should know why each of those numbers matters.

Benchmarking is the part of edge AI most often done wrong. The mistakes are predictable: timing the first iteration after a cold start, reporting only `mean`, forgetting to synchronize CUDA, measuring `model(x)` and calling that "the system latency", running the laptop on battery and not labeling it. This chapter builds the discipline that the rest of the course (and every project in `experiments/benchmark_results/`) relies on.

---

## 1. What "latency" actually means

A useful latency report has at minimum:

- **Mean** — average per-iter latency.
- **P50 (median)** — the typical case.
- **P95** — the bad-but-common case; this is the number that fails you in production.
- **P99** — the tail; matters most for safety-critical systems.
- **Std deviation** — how jittery the system is.

A single mean is misleading. A model that runs at mean 20 ms but P95 = 80 ms cannot be deployed at 30 FPS. A model that runs at mean 25 ms but P95 = 27 ms probably can.

Always also report:

- **Warm-up iterations** (discarded).
- **Measured iterations** (used for stats).
- **Device + runtime + dtype + input shape.**

These are not optional; without them the latency number means nothing.

---

## 2. Latency vs throughput vs FPS

Three closely related metrics that are *not* the same:

| Metric | Definition | When to use |
|---|---|---|
| **Latency** | Time from one input arriving to one output produced. Reported as percentiles | Real-time, latency-sensitive systems |
| **Throughput** | Number of inputs processed per unit time, under load. Often higher than `1/latency` because of batching and pipelining | Offline batch scoring, multi-stream camera setups |
| **FPS** | Frames per second the *whole pipeline* (capture + preprocess + infer + postprocess + output) can sustain | Camera systems, video streams |

For a real-time camera classifier:

- Model-only latency is interesting for optimization, but
- **FPS end-to-end is the metric that decides if the system works**.

In this course's benchmark template, both are reported separately.

---

## 3. Model-only vs end-to-end latency

These are different numbers. Reporting only the first one is the most common edge-AI misrepresentation.

```
model_only_latency       =  model(x) call alone
end_to_end_latency       =  camera_capture
                          + preprocessing
                          + model_inference
                          + postprocessing
                          + visualization_or_output
                          + communication_or_action_delay
```

For a 30-FPS USB webcam pipeline:
- Capture: ~30 ms (the camera limit; you can't go faster than the camera).
- Preprocess: ~3-5 ms (resize + normalize).
- Inference: ~10-50 ms (the part you optimize).
- Postprocess: ~3-5 ms.
- Output: ~5-15 ms (overlay + display, or MQTT publish).

Reporting "the model runs at 10 ms" when the whole pipeline runs at 50 ms is technically true but operationally misleading. Course rule: every benchmark in `experiments/benchmark_results/` reports both.

---

## 4. Memory: RSS and VRAM

Two numbers to track, both as **peak under steady load**:

- **CPU RSS (resident set size)** — total physical RAM your process consumes. The OS sees this; `top`, `ps`, and `psutil` report it.
- **GPU VRAM** — what your model and its activations occupy on the GPU. `nvidia-smi` and `torch.cuda.max_memory_allocated()` both report it, but watch the units (MB vs MiB).

Edge devices have hard memory ceilings:

| Device | RAM available to your process (approx) |
|---|---|
| Arduino Nano 33 BLE | ~256 KB |
| ESP32-S3 | ~512 KB |
| Raspberry Pi 5 (8 GB) | ~6 GB after OS |
| Jetson Orin Nano (8 GB) | ~6 GB shared CPU+GPU |
| Intel NUC (16-32 GB) | varies |

Knowing the model fits *and the activations also fit* is part of the design. The course benchmark uses `psutil` (CPU) + `torch.cuda.max_memory_allocated` (GPU) to track both.

---

## 5. Utilization, temperature, power

For honest edge benchmarks, you should also report (when measurable):

| Metric | Tool / source | Where it matters |
|---|---|---|
| CPU utilization | `psutil.cpu_percent()` | CPU-only edge devices |
| GPU utilization | `nvidia-smi`, `tegrastats` (Jetson) | Catches "model uses 5% of the GPU" mistakes |
| Temperature | `tegrastats`, `vcgencmd` (Pi), `lm_sensors` | Detects thermal throttling |
| Power | `tegrastats` (Jetson Orin), USB watt-meters, INA219 | Battery and energy budgets |

These are *recommended*, not strictly required, for laptop-class development. They become essential on Jetson, Pi, and TinyML devices.

---

## 6. The standard course benchmark report

Every project in `experiments/benchmark_results/` follows this schema (matches the metrics table in `Instruction.pdf` §14):

| Field | Type | Required? | Source |
|---|---|---|---|
| `model_size_mb` | float | yes | file size on disk |
| `latency_mean_ms` | float | yes | `benchmark_latency` |
| `latency_p50_ms` | float | yes | |
| `latency_p95_ms` | float | yes | |
| `latency_p99_ms` | float | recommended | |
| `fps_end_to_end` | float | yes with camera | `benchmark_fps` |
| `memory_peak_mb` | float | yes | `benchmark_memory` |
| `cpu_gpu_npu_utilization` | float | recommended | platform tools |
| `temperature_celsius` | float | recommended (essential on edge) | sensor |
| `power_watt` | float | device-dependent | sensor |

Also include metadata: `device`, `runtime`, `precision`, `input_shape`, `batch_size`, `warmup`, `repeat`, `hostname`, `os`, `python_version`, `torch_version`, `timestamp`.

---

## 7. The benchmark API in this repo

The course ships with a small Python package under `src/benchmark/`:

```python
from src.benchmark import (
    benchmark_latency,   # latency percentiles for any zero-arg callable
    benchmark_memory,    # peak CPU RSS + GPU VRAM during a workload
    benchmark_fps,       # end-to-end FPS for a streaming pipeline
    bench_full,          # combines all three with metadata
    format_report,       # pretty-print the report
    save_report_json,    # persist to experiments/benchmark_results/
)
```

Smallest possible usage:

```python
import torch
from torchvision import models
from src.benchmark import bench_full, format_report

model = models.mobilenet_v3_small(weights="DEFAULT").eval()
x = torch.randn(1, 3, 224, 224)

def step():
    with torch.no_grad():
        return model(x)

report = bench_full(step, name="mobilenetv3-cpu-fp32", device="cpu",
                    extras={"model": "mobilenet_v3_small", "params_millions": 2.54})
print(format_report(report))
```

Or as a CLI:

```bash
python -m src.benchmark --model mobilenet_v3_small --device cpu --pipeline
python -m src.benchmark --model resnet50 --device cuda --warmup 20 --repeat 100
```

The CLI saves a timestamped JSON to `experiments/benchmark_results/`. Drop those JSON files in your final-project report.

---

## 8. Reading the example report

A typical CPU run looks like:

```
=== benchmark: mobilenet_v3_small-cpu-fp32 ===
device=cpu  host=...  os=Linux...  cpus=16  torch=2.12

[ model-only latency ]
  warmup=20  repeat=100
  mean=4.81 ms  std=0.44
  P50=4.70  P95=5.47  P99=6.27
  fps_estimate=207.9

[ memory ]
  cpu_rss_peak=692.8 MB  (+0.5 MB over baseline)

[ end-to-end pipeline ]
  duration=2.00s  iters=394
  fps_end_to_end=196.9
  per-iter mean=5.08 ms  P50=4.86  P95=6.53  P99=8.76
```

How to read it:

- **fps_estimate (207.9)** is `1000 / mean_model_only_latency`. It's the *upper bound* — the FPS you would see if capture and postprocessing were free.
- **fps_end_to_end (196.9)** is the *actual* throughput including preprocessing and postprocessing. It is always lower than `fps_estimate`. The closer the two are, the less you can win by optimizing the model alone.
- **P95 (5.47 ms model-only)** is the tail you should plan for. If your latency budget is 33 ms (30 FPS), this is fine; if your budget is 5 ms, it's not.
- **cpu_rss_peak (692.8 MB)** is the total process footprint — most of that is the Python interpreter + torch + cuDNN + libraries, not the model itself. The `delta` column (+0.5 MB) is what the model added.

---

## 9. Common benchmark mistakes

In order of how often the course graders see them:

1. **No warm-up.** First call includes JIT / cuDNN / allocator overhead. Use ≥10 warm-up iters.
2. **Mean only.** Report at least mean + P50 + P95.
3. **CUDA without sync.** Without `torch.cuda.synchronize()` you measure launch latency, not compute.
4. **Model-only as if it were end-to-end.** Add capture + preprocess + postprocess.
5. **Unlabeled runs.** "It runs at 50 FPS" — on what hardware, runtime, precision, input size?
6. **Battery-on-laptop benchmark.** A throttled laptop reports half the real edge perf.
7. **Comparing across machines.** Same model on a Jetson and a laptop are *different runs*, label them.
8. **Reporting one number.** Always run the full benchmark; one number hides the failure modes.
9. **Skipping memory.** A model that fits on the laptop may not fit on the Pi.
10. **No JSON.** "I'll remember the numbers." You won't. Save them.

---

## 10. Recommended benchmark workflow

For every model you ship in this course:

1. Choose your representative input (one image of typical resolution; one frame from your camera).
2. Run the model-only benchmark with `warmup=20, repeat=100`. Save the JSON.
3. Wrap the same model in your real pipeline (capture / preprocess / postprocess) and run the FPS benchmark with `duration_s=10`. Save the JSON.
4. Track peak memory under sustained inference.
5. On thermally constrained devices (Jetson Orin Nano under 25 W, Pi without cooling, MCU), also run a *long* sustained-load test (≥5 min) and report whether latency drifts upward as the device heats.
6. Re-run after every optimization (Ch 8). The numbers in the report are the ground truth, not your hopes.

---

## 11. What you should be able to do after this chapter

- Use `src.benchmark` to produce a standard report for any callable inference function.
- Read the report and identify which P-tile or which stage is the bottleneck.
- Distinguish model-only latency from end-to-end latency in any project.
- Track peak CPU RSS and (where applicable) GPU VRAM.
- Save and version benchmark results as JSON in `experiments/benchmark_results/`.
- Complete the Chapter 4 assignment, which requires a benchmark across at least 2 of {batch size, device, model variant}.

---

## 12. Files produced by this chapter

- `docs/04_benchmarking_and_profiling.md` — this file.
- `src/benchmark/__init__.py` — package entrypoints.
- `src/benchmark/latency.py` — latency percentiles for any callable.
- `src/benchmark/memory.py` — CPU RSS + GPU VRAM peak.
- `src/benchmark/fps.py` — end-to-end FPS for streaming pipelines.
- `src/benchmark/report.py` — combined `bench_full` + format / save helpers.
- `src/benchmark/__main__.py` — CLI for torchvision classifiers.
- `assignments/assignment_03_latency_benchmark.md` — benchmark assignment.
