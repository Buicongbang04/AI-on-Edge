"""Latency benchmark — mean, P50/P95/P99, FPS estimate.

This is the canonical timing primitive for the whole course. Every later benchmark
script wraps this one.

Usage:
    from src.benchmark import benchmark_latency

    def step():
        return model(x)   # any callable, no args

    report = benchmark_latency(step, warmup=20, repeat=100, device="cuda")
    print(report["latency_p95_ms"])
"""
from __future__ import annotations

import time
from typing import Callable, Literal

import numpy as np

# torch import is optional — only needed when device == "cuda" for proper sync.
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def benchmark_latency(
    step: Callable[[], object],
    *,
    warmup: int = 20,
    repeat: int = 100,
    device: Literal["cpu", "cuda"] = "cpu",
) -> dict[str, float | int]:
    """Time a callable many times and return latency statistics.

    Args:
        step: a zero-arg callable that performs one inference. The callable should
              NOT include any per-call overhead unrelated to the work being measured.
        warmup: number of iterations to discard before measuring. Always include some
                warm-up for CUDA / cuDNN / oneDNN; 5-10 is enough for CPU, 10-20 for GPU.
        repeat: number of measured iterations.
        device: "cpu" or "cuda". On "cuda" the timer issues `torch.cuda.synchronize()`
                so the recorded time covers the actual GPU work.

    Returns:
        Dict with: warmup, repeat, device,
        latency_mean_ms, latency_std_ms,
        latency_p50_ms, latency_p95_ms, latency_p99_ms,
        latency_min_ms, latency_max_ms,
        fps_estimate (= 1000 / mean).

    Notes:
        - Use `time.perf_counter`, NOT `time.time()`.
        - CUDA: a synchronize is required before stopping the timer; otherwise the
          measurement reports kernel launch latency, not compute latency.
    """
    if warmup < 0 or repeat <= 0:
        raise ValueError("warmup must be >= 0 and repeat must be > 0")

    sync = _make_sync(device)

    # Warm-up — discard
    for _ in range(warmup):
        step()
        sync()

    times_ms: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        step()
        sync()
        times_ms.append((time.perf_counter() - start) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "warmup": warmup,
        "repeat": repeat,
        "device": device,
        "latency_mean_ms": float(arr.mean()),
        "latency_std_ms": float(arr.std(ddof=0)),
        "latency_p50_ms": float(np.percentile(arr, 50)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "latency_p99_ms": float(np.percentile(arr, 99)),
        "latency_min_ms": float(arr.min()),
        "latency_max_ms": float(arr.max()),
        "fps_estimate": float(1000.0 / max(arr.mean(), 1e-9)),
    }


def _make_sync(device: str) -> Callable[[], None]:
    """Return a (no-arg) sync function appropriate for the device."""
    if device == "cuda":
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available")
        return torch.cuda.synchronize  # type: ignore[return-value]
    # CPU — no-op
    return lambda: None
