"""OpenVINO inference reference script (Chapter 7).

This script works on **Intel CPUs by default**, and additionally targets Intel iGPU
or NPU when those devices are detected. The same model file runs on all three —
the only change is `--device {CPU,GPU,NPU,AUTO}`.

Typical workflow:
    1. Export the model to ONNX (Chapter 5):
        python src/export/export_onnx.py --model mobilenet_v3_small

    2. Run inference directly on the ONNX (OpenVINO can load ONNX):
        python src/inference/infer_openvino.py \\
            --model experiments/exported_models/mobilenet_v3_small.onnx \\
            --image datasets/sample.jpg

    3. (optional) Convert to OpenVINO IR for faster cold starts and INT8 quantization:
        python -c "import openvino as ov; \\
            ov.save_model(ov.convert_model('experiments/exported_models/mobilenet_v3_small.onnx'), \\
                          'experiments/exported_models/mobilenet_v3_small.xml')"

    4. Run via the IR:
        python src/inference/infer_openvino.py \\
            --model experiments/exported_models/mobilenet_v3_small.xml \\
            --image datasets/sample.jpg --device AUTO

Note:
    - If you do NOT have an Intel iGPU or NPU exposed, the only available device is "CPU".
    - "AUTO" picks the best device (NPU > GPU > CPU when present and model-compatible).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

try:
    import openvino as ov            # type: ignore[import-not-found]
    HAVE_OV = True
except ImportError:
    HAVE_OV = False


def _install_hint() -> str:
    return (
        "OpenVINO is not installed.\n"
        "  - Install: pip install openvino openvino-dev\n"
        "  - For iGPU on Linux you may also need: sudo apt install intel-opencl-icd\n"
        "  - For NPU (Core Ultra): install the Intel NPU driver per Intel docs.\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference via OpenVINO on Intel CPU / iGPU / NPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to .onnx or .xml (OpenVINO IR).")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="AUTO",
                        help="OpenVINO device: CPU, GPU, NPU, AUTO (picks best).")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    if not HAVE_OV:
        print(_install_hint())
        return 0  # not a hard failure

    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")

    from PIL import Image
    from torchvision.models import MobileNet_V3_Small_Weights

    preprocess = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    classes = MobileNet_V3_Small_Weights.IMAGENET1K_V1.meta["categories"]

    img = Image.open(args.image).convert("RGB")
    x = preprocess(img).unsqueeze(0).numpy().astype(np.float32)

    core = ov.Core()
    available = core.available_devices
    print(f"openvino available devices: {available}")
    if args.device != "AUTO" and args.device not in available:
        print(f"WARNING: device {args.device} not in available devices; "
              f"falling back to AUTO")
        args.device = "AUTO"

    print(f"loading {args.model} on device={args.device} ...")
    compiled = core.compile_model(str(args.model), device_name=args.device)
    infer_request = compiled.create_infer_request()
    input_port = compiled.inputs[0]

    # Warm-up
    for _ in range(args.warmup):
        infer_request.infer({input_port: x})

    # Timed
    times = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        result = infer_request.infer({input_port: x})
        times.append((time.perf_counter() - start) * 1000.0)

    logits = next(iter(result.values()))
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    top = probs.flatten().argsort()[::-1][:5]

    times = np.array(times, dtype=np.float64)
    print(f"\nlatency mean={times.mean():.2f} ms  "
          f"P50={np.percentile(times, 50):.2f}  P95={np.percentile(times, 95):.2f}  "
          f"P99={np.percentile(times, 99):.2f}  "
          f"fps={1000.0 / times.mean():.1f}")
    print("\ntop-5:")
    for i in top:
        print(f"  {probs.flatten()[i] * 100:6.2f}%  {classes[i]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
