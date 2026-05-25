"""Standard benchmark report — combines latency + memory + (optional) FPS.

Use `bench_full(...)` to produce one report dict that matches the metrics table
in the course (see `docs/04_benchmarking_and_profiling.md`).

Use `format_report(...)` for a printable summary, and `save_report_json(...)` to
persist results to `experiments/benchmark_results/`.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import time
from pathlib import Path
from typing import Callable, Literal

from .latency import benchmark_latency
from .memory import benchmark_memory
from .fps import benchmark_fps

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def bench_full(
    step: Callable[[], object],
    *,
    name: str,
    device: Literal["cpu", "cuda"] = "cpu",
    warmup: int = 20,
    repeat: int = 100,
    memory_iters: int = 50,
    pipeline_step: Callable[[], object] | None = None,
    pipeline_duration_s: float = 5.0,
    extras: dict | None = None,
) -> dict:
    """Run the standard course benchmark: latency + memory + (optional) FPS.

    Args:
        step: zero-arg callable for *model-only* latency and memory measurement.
        name: human-readable label for the run (e.g. "mobilenetv3_small-cpu-fp32").
        device: 'cpu' or 'cuda'.
        warmup, repeat: passed to `benchmark_latency`.
        memory_iters: passed to `benchmark_memory`.
        pipeline_step: optional zero-arg callable for *end-to-end* FPS measurement.
                       If provided, the pipeline benchmark also runs.
        pipeline_duration_s: how long the FPS benchmark runs.
        extras: optional dict of extra metadata to include in the report
                (e.g. model size, runtime, input resolution).

    Returns:
        A flat dict suitable for printing and JSON serialization.
    """
    started = time.time()

    lat = benchmark_latency(step, warmup=warmup, repeat=repeat, device=device)
    mem = benchmark_memory(step, iters=memory_iters, device=device)

    report: dict = {
        "name": name,
        "timestamp": int(started),
        "hostname": socket.gethostname(),
        "device": device,
        "system": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count() or 0,
        },
    }
    if torch is not None:
        report["system"]["torch_version"] = torch.__version__
        if device == "cuda" and torch.cuda.is_available():
            report["system"]["cuda_device"] = torch.cuda.get_device_name(0)

    report["latency"] = lat
    report["memory"] = mem

    if pipeline_step is not None:
        report["pipeline"] = benchmark_fps(pipeline_step, duration_s=pipeline_duration_s)

    if extras:
        report["extras"] = extras

    return report


def format_report(report: dict) -> str:
    """Pretty-print a benchmark report for stdout."""
    lat = report["latency"]
    mem = report["memory"]
    pipe = report.get("pipeline")
    extras = report.get("extras", {})
    sysinfo = report.get("system", {})

    lines = [
        f"=== benchmark: {report['name']} ===",
        f"device={report['device']}  host={report['hostname']}",
        f"os={sysinfo.get('os', 'unknown')}  python={sysinfo.get('python', '?')}  cpus={sysinfo.get('cpu_count', '?')}",
    ]
    if torch_v := sysinfo.get("torch_version"):
        lines.append(f"torch={torch_v}")
    if cuda_dev := sysinfo.get("cuda_device"):
        lines.append(f"cuda_device={cuda_dev}")
    if extras:
        lines.append("extras: " + ", ".join(f"{k}={v}" for k, v in extras.items()))

    lines += [
        "",
        "[ model-only latency ]",
        f"  warmup={lat['warmup']}  repeat={lat['repeat']}",
        f"  mean={lat['latency_mean_ms']:.2f} ms  std={lat['latency_std_ms']:.2f}",
        f"  P50={lat['latency_p50_ms']:.2f}  P95={lat['latency_p95_ms']:.2f}  P99={lat['latency_p99_ms']:.2f}",
        f"  min={lat['latency_min_ms']:.2f}  max={lat['latency_max_ms']:.2f}",
        f"  fps_estimate={lat['fps_estimate']:.1f}",
        "",
        "[ memory ]",
        f"  cpu_rss_peak={mem['cpu_rss_peak_mb']:.1f} MB  (+{mem['cpu_rss_delta_mb']:.1f} MB over baseline)",
    ]
    if "gpu_vram_peak_mb" in mem:
        lines.append(
            f"  gpu_vram_peak={mem['gpu_vram_peak_mb']:.1f} MB  (+{mem['gpu_vram_delta_mb']:.1f} MB over baseline)"
        )

    if pipe is not None:
        lines += [
            "",
            "[ end-to-end pipeline ]",
            f"  duration={pipe['duration_s']:.2f}s  iters={pipe['iters']}",
            f"  fps_end_to_end={pipe['fps_end_to_end']:.1f}",
            f"  per-iter mean={pipe['latency_per_iter_mean_ms']:.2f} ms"
            f"  P50={pipe['latency_per_iter_p50_ms']:.2f}"
            f"  P95={pipe['latency_per_iter_p95_ms']:.2f}"
            f"  P99={pipe['latency_per_iter_p99_ms']:.2f}",
        ]
    return "\n".join(lines)


def save_report_json(report: dict, out_dir: str | Path = "experiments/benchmark_results") -> Path:
    """Save a report dict to a timestamped JSON file. Returns the file path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = report.get("name", "report")
    ts = report.get("timestamp", int(time.time()))
    path = out_dir / f"{name}-{ts}.json"
    path.write_text(json.dumps(report, indent=2))
    return path
