"""Benchmark utilities for Edge AI.

Three pieces:
- `latency.py`    — latency percentiles (P50/P95/P99) for any callable.
- `memory.py`     — peak CPU RSS (and GPU VRAM if available).
- `fps.py`        — end-to-end FPS for a streaming pipeline (capture → infer → postprocess).

Combine them with `bench_full(...)` for the standard benchmark report used everywhere in the course.
"""
from .latency import benchmark_latency
from .memory import benchmark_memory
from .fps import benchmark_fps
from .report import bench_full, format_report, save_report_json

__all__ = [
    "benchmark_latency",
    "benchmark_memory",
    "benchmark_fps",
    "bench_full",
    "format_report",
    "save_report_json",
]
