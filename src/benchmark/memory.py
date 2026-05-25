"""Memory benchmark — peak CPU RSS and (optionally) peak GPU VRAM.

Sampling-based: spawns a tiny sidecar thread that polls process RSS via psutil
during the benchmark window, and reads `torch.cuda.max_memory_allocated()` on CUDA.

Usage:
    from src.benchmark import benchmark_memory

    def step():
        return model(x)

    report = benchmark_memory(step, iters=50, device="cpu", poll_hz=20)
    print(report["cpu_rss_peak_mb"])
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Literal

import psutil

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class _RSSPoller:
    """Background thread that records peak RSS of the current process."""
    poll_hz: float = 20.0
    _stop: threading.Event = None  # type: ignore[assignment]
    _thread: threading.Thread = None  # type: ignore[assignment]
    peak_bytes: int = 0
    samples: int = 0

    def start(self) -> None:
        self._stop = threading.Event()
        proc = psutil.Process(os.getpid())
        self.peak_bytes = proc.memory_info().rss
        interval = 1.0 / max(self.poll_hz, 1.0)

        def _loop() -> None:
            while not self._stop.is_set():
                rss = proc.memory_info().rss
                if rss > self.peak_bytes:
                    self.peak_bytes = rss
                self.samples += 1
                self._stop.wait(interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def benchmark_memory(
    step: Callable[[], object],
    *,
    iters: int = 50,
    device: Literal["cpu", "cuda"] = "cpu",
    poll_hz: float = 20.0,
) -> dict[str, float | int]:
    """Measure peak memory while running `step` `iters` times.

    Args:
        step: zero-arg callable performing one inference.
        iters: number of iterations to run (more iters = more reliable peak).
        device: 'cpu' (RSS only) or 'cuda' (RSS + VRAM).
        poll_hz: polling rate for CPU RSS in Hz.

    Returns:
        Dict with iters, device,
        cpu_rss_baseline_mb, cpu_rss_peak_mb, cpu_rss_delta_mb,
        and on CUDA also: gpu_vram_baseline_mb, gpu_vram_peak_mb, gpu_vram_delta_mb.
    """
    if iters <= 0:
        raise ValueError("iters must be > 0")

    proc = psutil.Process(os.getpid())
    rss_baseline = proc.memory_info().rss

    gpu_baseline = 0
    if device == "cuda":
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gpu_baseline = torch.cuda.memory_allocated()

    poller = _RSSPoller(poll_hz=poll_hz)
    poller.start()
    try:
        # A short warm-up so memory allocations stabilize
        for _ in range(5):
            step()
        # Measured loop
        for _ in range(iters):
            step()
            if device == "cuda":
                torch.cuda.synchronize()
        # tiny tail so the poller has time to capture the final state
        time.sleep(1.0 / max(poll_hz, 1.0))
    finally:
        poller.stop()

    out: dict[str, float | int] = {
        "iters": iters,
        "device": device,
        "cpu_rss_baseline_mb": rss_baseline / 1024 / 1024,
        "cpu_rss_peak_mb": poller.peak_bytes / 1024 / 1024,
        "cpu_rss_delta_mb": (poller.peak_bytes - rss_baseline) / 1024 / 1024,
        "cpu_rss_samples": poller.samples,
    }
    if device == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated()
        out.update({
            "gpu_vram_baseline_mb": gpu_baseline / 1024 / 1024,
            "gpu_vram_peak_mb": peak_alloc / 1024 / 1024,
            "gpu_vram_delta_mb": (peak_alloc - gpu_baseline) / 1024 / 1024,
        })
    return out
