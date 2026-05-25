"""Model optimization helpers (Chapter 8).

- `quantization.quantize_onnx_dynamic`  — INT8 dynamic quantization (weights only)
- `quantization.quantize_onnx_static`   — INT8 static quantization (weights + activations, needs calibration)
- `quantization.compare_models`          — accuracy + size + latency compare across models
"""
from .quantization import (
    quantize_onnx_dynamic,
    quantize_onnx_static,
    compare_models,
    RandomImageCalibrationDataReader,
)

__all__ = [
    "quantize_onnx_dynamic",
    "quantize_onnx_static",
    "compare_models",
    "RandomImageCalibrationDataReader",
]
