"""JSONL device-log writer for edge inference events (Chapter 19).

Schema matches the Instruction.pdf §15 (device_log_schema) plus a few course extras.
Each call to `log()` appends one JSON object per line.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


# The standard course schema (matches Instruction.pdf section 15)
INFERENCE_LOG_SCHEMA = {
    "timestamp":          "float",
    "device_id":          "str",
    "model_version":      "str",
    "runtime":            "str",
    "input_source":       "str",     # 'camera' | 'sensor' | 'file' | ...
    "latency_ms":         "float",
    "confidence":         "float",   # 0..1
    "prediction":         "str",     # short text or class label
    "action_command":     "str|null",
    "temperature_celsius":"float|null",
    "error_code":         "str|null",
    "fallback_triggered": "bool",
}


class DeviceLogger:
    """Thread-safe JSONL logger for per-inference events.

    Example:
        logger = DeviceLogger("experiments/logs/device_42.jsonl",
                              device_id="device-42",
                              model_version="mobilenetv3-small@v1.0.3",
                              runtime="onnxruntime-CPU")
        logger.log(latency_ms=8.5, confidence=0.92, prediction="cat",
                   input_source="camera")
    """

    def __init__(
        self,
        path: str | Path,
        *,
        device_id: str,
        model_version: str,
        runtime: str,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._defaults = {
            "device_id": device_id,
            "model_version": model_version,
            "runtime": runtime,
        }
        self._lock = threading.Lock()
        self._fp = self.path.open("a", buffering=1)  # line-buffered

    def log(
        self,
        *,
        latency_ms: float,
        confidence: float,
        prediction: str,
        input_source: str = "camera",
        action_command: str | None = None,
        temperature_celsius: float | None = None,
        error_code: str | None = None,
        fallback_triggered: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "timestamp": time.time(),
            **self._defaults,
            "input_source": input_source,
            "latency_ms": float(latency_ms),
            "confidence": float(confidence),
            "prediction": str(prediction),
            "action_command": action_command,
            "temperature_celsius": temperature_celsius,
            "error_code": error_code,
            "fallback_triggered": bool(fallback_triggered),
        }
        if extra:
            record.update(extra)
        line = json.dumps(record, default=str)
        with self._lock:
            self._fp.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            if not self._fp.closed:
                self._fp.close()

    def __enter__(self) -> "DeviceLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
