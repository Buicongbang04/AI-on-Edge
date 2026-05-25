"""ONNX INT8 quantization helpers (Chapter 8).

Two paths supported:

1. **Dynamic quantization** (`quantize_onnx_dynamic`):
   - Weights are quantized at export time, activations at runtime.
   - No calibration data needed.
   - Easiest path; modest speedup; some accuracy loss.

2. **Static quantization** (`quantize_onnx_static`):
   - Weights AND activations quantized at export time.
   - Requires a *calibration dataset* (a handful of representative inputs) so
     the quantizer can estimate activation ranges.
   - Larger speedup; sometimes more accuracy loss; needed for INT8 NPU/TPU paths.

Both produce a new `.onnx` file that runs in ONNX Runtime with the same API. See
`docs/08_model_optimization.md` for when to use which.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    QuantType,
    CalibrationDataReader,
    quantize_dynamic,
    quantize_static,
)


# ---------------------------------------------------------------------------
# Dynamic quantization
# ---------------------------------------------------------------------------

def quantize_onnx_dynamic(
    src: Path,
    dst: Path,
    *,
    weight_type: QuantType = QuantType.QInt8,
) -> Path:
    """Quantize weights of an ONNX model to INT8.

    Args:
        src: input .onnx file
        dst: output .onnx file
        weight_type: QuantType.QInt8 or QuantType.QUInt8 (most CPU EPs prefer QInt8)

    Returns:
        Path to the quantized model.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=src.as_posix(),
        model_output=dst.as_posix(),
        weight_type=weight_type,
    )
    return dst


# ---------------------------------------------------------------------------
# Static quantization — calibration data reader
# ---------------------------------------------------------------------------

class RandomImageCalibrationDataReader(CalibrationDataReader):
    """A trivial calibration reader that yields N random normal inputs.

    Good enough for a teaching demo on synthetic data. For production, swap this
    with a reader that returns real preprocessed images from your validation set.

    Usage:
        reader = RandomImageCalibrationDataReader(
            input_name="input", shape=(1, 3, 224, 224), num_samples=32,
        )
        quantize_static(src, dst, reader)
    """

    def __init__(
        self,
        input_name: str,
        shape: tuple[int, ...],
        num_samples: int = 32,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self._iter = iter([
            {input_name: rng.standard_normal(shape, dtype=np.float32)}
            for _ in range(num_samples)
        ])

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iter, None)


def quantize_onnx_static(
    src: Path,
    dst: Path,
    calibration_reader: CalibrationDataReader,
    *,
    activation_type: QuantType = QuantType.QInt8,
    weight_type: QuantType = QuantType.QInt8,
) -> Path:
    """Quantize weights AND activations to INT8 via a calibration dataset."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=src.as_posix(),
        model_output=dst.as_posix(),
        calibration_data_reader=calibration_reader,
        activation_type=activation_type,
        weight_type=weight_type,
    )
    return dst


# ---------------------------------------------------------------------------
# Compare two or more ONNX models
# ---------------------------------------------------------------------------

def _file_size_mb(p: Path) -> float:
    return p.stat().st_size / 1024 / 1024


def compare_models(
    models: dict[str, Path],
    *,
    sample_inputs: list[np.ndarray] | None = None,
    input_name: str | None = None,
    reference_label: str | None = None,
    repeat: int = 30,
    warmup: int = 5,
    providers: Iterable[str] = ("CPUExecutionProvider",),
    metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> list[dict]:
    """Compare several ONNX models on (size, latency, optional accuracy).

    Args:
        models: dict mapping label → onnx file path.
        sample_inputs: list of input arrays. If None, uses one random batch.
        input_name: name of the input tensor (auto-detected from first model if None).
        reference_label: which label is the reference (for max-abs-diff against others).
        repeat, warmup: passed to a small inline latency loop.
        providers: ONNX Runtime providers list.
        metric_fn(reference_out, candidate_out) → float, e.g. argmax agreement rate.

    Returns:
        List of dicts: label, size_mb, latency_p50_ms, latency_p95_ms, fps,
                       and (if reference_label provided) max_abs_diff, metric.
    """
    import time

    if not models:
        return []
    if sample_inputs is None:
        # default: one synthetic batch shaped like a standard ImageNet input
        sample_inputs = [np.random.randn(1, 3, 224, 224).astype(np.float32)]

    # Build sessions
    sessions: dict[str, ort.InferenceSession] = {}
    in_names: dict[str, str] = {}
    for label, path in models.items():
        sess = ort.InferenceSession(path.as_posix(), providers=list(providers))
        sessions[label] = sess
        in_names[label] = input_name or sess.get_inputs()[0].name

    # Reference outputs for accuracy comparison
    ref_outputs: list[np.ndarray] | None = None
    if reference_label and reference_label in sessions:
        ref_outputs = [
            sessions[reference_label].run(None, {in_names[reference_label]: x})[0]
            for x in sample_inputs
        ]

    rows: list[dict] = []
    for label, sess in sessions.items():
        # Warm-up + latency loop on the first sample input
        x = sample_inputs[0]
        for _ in range(warmup):
            sess.run(None, {in_names[label]: x})
        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            sess.run(None, {in_names[label]: x})
            times.append((time.perf_counter() - start) * 1000.0)
        arr = np.array(times, dtype=np.float64)

        row: dict = {
            "label": label,
            "size_mb": _file_size_mb(models[label]),
            "latency_mean_ms": float(arr.mean()),
            "latency_p50_ms": float(np.percentile(arr, 50)),
            "latency_p95_ms": float(np.percentile(arr, 95)),
            "fps_estimate": float(1000.0 / arr.mean()),
        }

        # Accuracy comparison vs reference (max abs diff + metric)
        if ref_outputs is not None and label != reference_label:
            diffs: list[float] = []
            metric_vals: list[float] = []
            for x, ref_y in zip(sample_inputs, ref_outputs):
                cand_y = sess.run(None, {in_names[label]: x})[0]
                diffs.append(float(np.abs(cand_y - ref_y).max()))
                if metric_fn is not None:
                    metric_vals.append(float(metric_fn(ref_y, cand_y)))
            row["max_abs_diff_vs_ref"] = max(diffs)
            if metric_vals:
                row["metric_vs_ref"] = float(np.mean(metric_vals))

        rows.append(row)

    return rows


def argmax_agreement(ref: np.ndarray, cand: np.ndarray) -> float:
    """Fraction of items where argmax agrees between reference and candidate."""
    return float((ref.argmax(axis=-1) == cand.argmax(axis=-1)).mean())
