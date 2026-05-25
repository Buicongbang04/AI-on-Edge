"""EdgeOps helpers (Chapter 19).

- `logging.DeviceLogger`             — JSONL log writer for per-inference events
- `versioning.compute_model_checksum` — SHA256 of a model file, used in deployment
- `versioning.ModelVersion`           — model_version + runtime_version + input_shape + preprocessing_config + checksum
- `fallback.confidence_fallback`     — if model confidence < threshold, return a safe default
"""
from .logging import DeviceLogger, INFERENCE_LOG_SCHEMA
from .versioning import ModelVersion, compute_model_checksum
from .fallback import confidence_fallback, FallbackAction

__all__ = [
    "DeviceLogger",
    "INFERENCE_LOG_SCHEMA",
    "ModelVersion",
    "compute_model_checksum",
    "confidence_fallback",
    "FallbackAction",
]
