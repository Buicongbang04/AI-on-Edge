"""PyTorch inference reference script (Chapter 3).

Loads a pretrained image classifier from torchvision, runs inference on a single
image or a folder of images, and prints top-5 predictions with confidence and
a basic per-image latency.

This is the *reference* inference script for the course. Later chapters add:
- proper benchmarking (Ch 4)
- ONNX export and ONNX Runtime inference (Ch 5-6)
- INT8 quantization (Ch 8)
- camera loop (Ch 9)

Example:
    python src/inference/infer_pytorch.py --image datasets/sample.jpg
    python src/inference/infer_pytorch.py --image-dir datasets/test_images
    python src/inference/infer_pytorch.py --image datasets/sample.jpg --device cuda
    python src/inference/infer_pytorch.py --image datasets/sample.jpg --model resnet50
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Models supported out of the box (all pretrained on ImageNet)
# ---------------------------------------------------------------------------
MODEL_FACTORY = {
    "mobilenet_v3_small": (
        models.mobilenet_v3_small,
        models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    ),
    "mobilenet_v3_large": (
        models.mobilenet_v3_large,
        models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
    ),
    "resnet18": (
        models.resnet18,
        models.ResNet18_Weights.IMAGENET1K_V1,
    ),
    "resnet50": (
        models.resnet50,
        models.ResNet50_Weights.IMAGENET1K_V2,
    ),
}


def load_model(name: str, device: torch.device) -> tuple[torch.nn.Module, transforms.Compose, list[str]]:
    """Load a pretrained classifier, its preprocessing transform, and class names."""
    if name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(MODEL_FACTORY)}")

    builder, weights = MODEL_FACTORY[name]
    model = builder(weights=weights)
    model.eval()
    model.to(device)

    preprocess = weights.transforms()
    classes = weights.meta["categories"]
    return model, preprocess, classes


def iter_image_paths(image: Path | None, image_dir: Path | None) -> Iterable[Path]:
    if image is not None:
        yield image
    if image_dir is not None:
        for p in sorted(image_dir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                yield p


def infer_one(
    model: torch.nn.Module,
    preprocess: transforms.Compose,
    classes: list[str],
    image_path: Path,
    device: torch.device,
    topk: int = 5,
) -> tuple[list[tuple[str, float]], float]:
    """Run inference on a single image. Returns (top-k predictions, latency_ms)."""
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    # IMPORTANT for deployment-style inference:
    #   - model.eval()  : already done in load_model — disables dropout/batchnorm updates.
    #   - torch.no_grad(): disables autograd, large memory and small latency win.
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
    # Force sync on CUDA so latency includes the actual GPU work.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    top_probs, top_idx = probs.topk(topk, dim=1)
    top_probs = top_probs.squeeze(0).tolist()
    top_idx = top_idx.squeeze(0).tolist()
    return [(classes[i], p) for i, p in zip(top_idx, top_probs)], elapsed_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a pretrained PyTorch classifier on an image or folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=Path, default=None,
                        help="Path to a single image.")
    parser.add_argument("--image-dir", type=Path, default=None,
                        help="Path to a folder of images.")
    parser.add_argument("--model", choices=sorted(MODEL_FACTORY), default="mobilenet_v3_small",
                        help="Which torchvision model to load.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                        help="Inference device. 'auto' picks CUDA if available.")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warm-up iterations on the first image before timing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.image is None and args.image_dir is None:
        raise SystemExit("Provide --image and/or --image-dir.")

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cuda" if args.device == "cuda" else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False.")

    print(f"loading model={args.model} on device={device} ...")
    model, preprocess, classes = load_model(args.model, device)
    print(f"model loaded. {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params.")

    paths = list(iter_image_paths(args.image, args.image_dir))
    if not paths:
        raise SystemExit("No images found.")

    # Warm-up (using the first image) so the first reported latency is not dominated by
    # one-time setup costs (kernel JIT, memory alloc, cuDNN heuristics).
    print(f"warming up for {args.warmup} iteration(s) on {paths[0].name} ...")
    for _ in range(args.warmup):
        infer_one(model, preprocess, classes, paths[0], device, topk=args.topk)

    # Real inference
    print(f"\nrunning inference on {len(paths)} image(s):")
    latencies: list[float] = []
    for p in paths:
        preds, ms = infer_one(model, preprocess, classes, p, device, topk=args.topk)
        latencies.append(ms)
        print(f"\n[{p.name}]  latency={ms:.2f} ms")
        for cls, prob in preds:
            print(f"  {prob * 100:6.2f}%  {cls}")

    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"\nsummary: {len(latencies)} images, avg latency = {avg:.2f} ms")
        print("Note: this is *model-only* per-image latency. End-to-end pipeline latency")
        print("      (including image loading, preprocessing) is measured in Chapter 4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
