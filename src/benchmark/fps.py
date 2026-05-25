"""End-to-end FPS benchmark for streaming pipelines.

Unlike `benchmark_latency`, this measures the *whole* pipeline:
    capture → preprocess → infer → postprocess → output

The way to use it: pass a callable that performs ONE full pipeline iteration
(including any I/O), and an optional generator of inputs. The reported FPS is the
end-to-end rate, not the model-only rate. This is the only FPS number that has
operational meaning for camera systems.

Usage:
    from src.benchmark import benchmark_fps

    def pipeline_step():
        frame = capture()                          # USB camera, ~5-30 ms
        x = preprocess(frame)                      # resize + normalize
        y = run_inference(x)                       # model
        result = postprocess(y)                    # NMS / argmax / overlay
        emit(result)                               # display / log / MQTT
        return result

    report = benchmark_fps(pipeline_step, duration_s=10.0)
    print(report["fps_end_to_end"])
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np


def benchmark_fps(
    pipeline_step: Callable[[], object],
    *,
    duration_s: float = 10.0,
    max_iters: int = 100_000,
    warmup_iters: int = 5,
) -> dict[str, float | int]:
    """Run `pipeline_step` for `duration_s` seconds and report end-to-end FPS.

    The pipeline_step callable should perform ONE complete pipeline iteration —
    capture, preprocess, infer, postprocess, output. The FPS reported is then the
    end-to-end throughput, not the model-only throughput.

    Args:
        pipeline_step: zero-arg callable that performs one pipeline iteration.
        duration_s: how long to run the measured loop.
        max_iters: safety cap on iterations (in case duration_s is very long).
        warmup_iters: number of iterations to discard before timing starts.

    Returns:
        Dict with duration_s, iters,
        fps_end_to_end, latency_per_iter_mean_ms,
        latency_per_iter_p50_ms, latency_per_iter_p95_ms, latency_per_iter_p99_ms.

    Notes:
        - For camera pipelines, sub-30 FPS reports often hide *which stage* is the
          bottleneck. Use the latency benchmark on each stage in isolation to find
          out, then come back to FPS to validate the full pipeline.
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")

    # Warm-up
    for _ in range(warmup_iters):
        pipeline_step()

    per_iter: list[float] = []
    deadline = time.perf_counter() + duration_s
    iters = 0
    loop_start = time.perf_counter()
    while iters < max_iters:
        if time.perf_counter() >= deadline:
            break
        it_start = time.perf_counter()
        pipeline_step()
        per_iter.append((time.perf_counter() - it_start) * 1000.0)
        iters += 1
    loop_elapsed = time.perf_counter() - loop_start

    if iters == 0:
        raise RuntimeError("no iterations completed; raise duration_s")

    arr = np.asarray(per_iter, dtype=np.float64)
    return {
        "duration_s": loop_elapsed,
        "iters": iters,
        "fps_end_to_end": iters / loop_elapsed,
        "latency_per_iter_mean_ms": float(arr.mean()),
        "latency_per_iter_p50_ms": float(np.percentile(arr, 50)),
        "latency_per_iter_p95_ms": float(np.percentile(arr, 95)),
        "latency_per_iter_p99_ms": float(np.percentile(arr, 99)),
    }
