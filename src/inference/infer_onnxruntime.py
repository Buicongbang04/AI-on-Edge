"""ONNX Runtime inference reference script (Chapter 6).

Loads a `.onnx` model exported in Chapter 5, runs inference on a single image
or a folder, and prints top-5 predictions plus basic latency. Mirrors
`infer_pytorch.py` so the two scripts can be compared directly.

Examples:
    python src/inference/infer_onnxruntime.py \
        --model experiments/exported_models/mobilenet_v3_small.onnx \
        --image datasets/sample.jpg

    python src/inference/infer_onnxruntime.py \
        --model experiments/exported_models/mobilenet_v3_small.onnx \
        --image-dir datasets/test_images --provider CUDAExecutionProvider
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights


# ImageNet preprocessing (matches MobileNetV3-Small training preprocessing).
# We reuse torchvision's transform but it returns a torch tensor — we then go to
# numpy because ONNX Runtime expects numpy arrays.
_IMAGENET_PREP = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
_IMAGENET_CLASSES = MobileNet_V3_Small_Weights.IMAGENET1K_V1.meta["categories"]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def iter_image_paths(image: Path | None, image_dir: Path | None) -> Iterable[Path]:
    if image is not None:
        yield image
    if image_dir is not None:
        for p in sorted(image_dir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                yield p


def load_session(model_path: Path, provider: str) -> ort.InferenceSession:
    available = ort.get_available_providers()
    if provider not in available:
        raise SystemExit(
            f"Provider {provider!r} is not available on this machine.\n"
            f"Available: {available}\n"
            f"Fix: install onnxruntime-gpu / openvino, or pick a provider from the list."
        )
    sess_opts = ort.SessionOptions()
    # Standard sensible defaults for edge inference
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 0  # 0 = let ORT decide based on cores
    sess = ort.InferenceSession(
        model_path.as_posix(),
        sess_options=sess_opts,
        providers=[provider],
    )
    return sess


def infer_one(
    sess: ort.InferenceSession,
    image_path: Path,
    topk: int = 5,
) -> tuple[list[tuple[str, float]], float]:
    """Run inference on a single image. Returns (top-k predictions, latency_ms)."""
    img = Image.open(image_path).convert("RGB")
    # Preprocess (torchvision tensor → numpy)
    x = _IMAGENET_PREP(img).unsqueeze(0).numpy()

    in_name = sess.get_inputs()[0].name
    start = time.perf_counter()
    (logits,) = sess.run(None, {in_name: x})
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    probs = softmax(logits, axis=1).squeeze(0)
    top_idx = probs.argsort()[::-1][:topk]
    return [(_IMAGENET_CLASSES[i], float(probs[i])) for i in top_idx], elapsed_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a torchvision-trained ONNX classifier via ONNX Runtime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to .onnx file (from Chapter 5).")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--provider", default="CPUExecutionProvider",
                        help="ONNX Runtime execution provider "
                             "(CPUExecutionProvider, CUDAExecutionProvider, "
                             "TensorrtExecutionProvider, OpenVINOExecutionProvider).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.image is None and args.image_dir is None:
        raise SystemExit("Provide --image and/or --image-dir.")
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}. Run src/export/export_onnx.py first.")

    print(f"loading {args.model} (provider={args.provider}) ...")
    sess = load_session(args.model, args.provider)

    paths = list(iter_image_paths(args.image, args.image_dir))
    if not paths:
        raise SystemExit("No images found.")

    print(f"warming up for {args.warmup} iteration(s) on {paths[0].name} ...")
    for _ in range(args.warmup):
        infer_one(sess, paths[0], topk=args.topk)

    print(f"\nrunning inference on {len(paths)} image(s):")
    latencies: list[float] = []
    for p in paths:
        preds, ms = infer_one(sess, p, topk=args.topk)
        latencies.append(ms)
        print(f"\n[{p.name}]  latency={ms:.2f} ms")
        for cls, prob in preds:
            print(f"  {prob * 100:6.2f}%  {cls}")

    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"\nsummary: {len(latencies)} images, avg latency = {avg:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
