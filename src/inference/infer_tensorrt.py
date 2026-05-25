"""TensorRT inference reference script (Chapter 7).

This script is for **NVIDIA Jetson or NVIDIA desktop GPU with TensorRT installed**.
On a laptop without TensorRT, the script exits cleanly with an install hint.

Typical workflow:
    1. Export the model to ONNX (Chapter 5):
        python src/export/export_onnx.py --model mobilenet_v3_small

    2. Build a TensorRT engine from the ONNX with `trtexec` (ships with TensorRT):
        trtexec --onnx=experiments/exported_models/mobilenet_v3_small.onnx \\
                --saveEngine=experiments/exported_models/mobilenet_fp16.engine \\
                --fp16

    3. Run inference on the engine:
        python src/inference/infer_tensorrt.py \\
            --engine experiments/exported_models/mobilenet_fp16.engine \\
            --image datasets/sample.jpg

Notes:
    - Engines are GPU-specific. An engine built on an RTX card will NOT load on a Jetson.
    - First call after engine load is the slowest (one-time setup). Always warm-up.
    - For INT8 you need to provide a calibration dataset; see Chapter 8.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

try:
    import tensorrt as trt           # type: ignore[import-not-found]
    import pycuda.driver as cuda     # type: ignore[import-not-found]
    import pycuda.autoinit  # noqa: F401   # type: ignore[import-not-found]
    HAVE_TRT = True
except ImportError:
    HAVE_TRT = False


def _install_hint() -> str:
    return (
        "TensorRT is not installed on this machine.\n"
        "  - On NVIDIA Jetson: TensorRT ships with JetPack — install/upgrade via SDK Manager.\n"
        "  - On desktop CUDA: install via NVIDIA package or pip (`pip install tensorrt`).\n"
        "  - You also need `pycuda` (or `cuda-python`) to drive the GPU buffers.\n"
    )


def load_engine(path: Path) -> "trt.ICudaEngine":
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"failed to deserialize {path}")
    return engine


def infer_engine(
    engine: "trt.ICudaEngine",
    x: np.ndarray,
    *,
    warmup: int = 5,
    repeat: int = 20,
) -> tuple[np.ndarray, dict[str, float]]:
    """Run inference repeatedly and return (output, latency_stats_ms)."""
    context = engine.create_execution_context()
    stream = cuda.Stream()

    # Allocate host + device buffers for every binding
    bindings: list[int] = []
    host_inputs: list[np.ndarray] = []
    host_outputs: list[np.ndarray] = []
    device_inputs: list[cuda.DeviceAllocation] = []
    device_outputs: list[cuda.DeviceAllocation] = []

    for i in range(engine.num_bindings):
        shape = engine.get_binding_shape(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        size = int(np.prod(shape))
        host_buf = np.empty(size, dtype=dtype).reshape(shape)
        dev_buf = cuda.mem_alloc(host_buf.nbytes)
        bindings.append(int(dev_buf))
        if engine.binding_is_input(i):
            host_inputs.append(host_buf)
            device_inputs.append(dev_buf)
        else:
            host_outputs.append(host_buf)
            device_outputs.append(dev_buf)

    # Place input data
    np.copyto(host_inputs[0], x.reshape(host_inputs[0].shape))

    def _one_iter() -> None:
        cuda.memcpy_htod_async(device_inputs[0], host_inputs[0], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], device_outputs[0], stream)
        stream.synchronize()

    # Warm-up
    for _ in range(warmup):
        _one_iter()

    # Timed
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        _one_iter()
        times.append((time.perf_counter() - start) * 1000.0)

    times = np.array(times, dtype=np.float64)
    return host_outputs[0], {
        "latency_mean_ms": float(times.mean()),
        "latency_p50_ms": float(np.percentile(times, 50)),
        "latency_p95_ms": float(np.percentile(times, 95)),
        "latency_p99_ms": float(np.percentile(times, 99)),
        "fps_estimate": float(1000.0 / times.mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on a TensorRT engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--engine", type=Path, required=True,
                        help="Path to a serialized TensorRT engine (.engine / .plan).")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    if not HAVE_TRT:
        print(_install_hint())
        return 0  # not a failure on machines without TRT

    args = parse_args()
    if not args.engine.exists():
        raise SystemExit(f"engine not found: {args.engine}")

    # Reuse the torchvision ImageNet preprocessing (consistent with Ch 3 / Ch 6)
    from PIL import Image
    from torchvision.models import MobileNet_V3_Small_Weights

    preprocess = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    classes = MobileNet_V3_Small_Weights.IMAGENET1K_V1.meta["categories"]

    img = Image.open(args.image).convert("RGB")
    x = preprocess(img).unsqueeze(0).numpy().astype(np.float32)

    print(f"loading engine: {args.engine}")
    engine = load_engine(args.engine)

    print(f"running inference (warmup={args.warmup}, repeat={args.repeat})")
    logits, stats = infer_engine(engine, x, warmup=args.warmup, repeat=args.repeat)

    probs = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum(axis=-1, keepdims=True)
    top = probs.flatten().argsort()[::-1][:5]
    print(f"\nlatency mean={stats['latency_mean_ms']:.2f} ms  "
          f"P50={stats['latency_p50_ms']:.2f}  P95={stats['latency_p95_ms']:.2f}  "
          f"P99={stats['latency_p99_ms']:.2f}  fps={stats['fps_estimate']:.1f}")
    print("\ntop-5:")
    for i in top:
        print(f"  {probs.flatten()[i] * 100:6.2f}%  {classes[i]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
