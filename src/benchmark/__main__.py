"""CLI for the benchmark module.

Runs the standard course benchmark on MobileNetV3-Small (or another torchvision
model), and saves a JSON report to `experiments/benchmark_results/`.

Examples:
    python -m src.benchmark --model mobilenet_v3_small --device cpu
    python -m src.benchmark --model resnet50 --device cuda
    python -m src.benchmark --model mobilenet_v3_small --device cpu --pipeline
"""
from __future__ import annotations

import argparse

import torch
from torchvision import models

from .report import bench_full, format_report, save_report_json


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standard benchmark for a torchvision classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODEL_FACTORY), default="mobilenet_v3_small")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--input-size", type=int, default=224,
                        help="Input image size (square). 224 for ImageNet models.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--memory-iters", type=int, default=50)
    parser.add_argument("--pipeline", action="store_true",
                        help="Also run a synthetic end-to-end pipeline benchmark "
                             "(preprocess + infer + argmax).")
    parser.add_argument("--pipeline-duration", type=float, default=5.0)
    parser.add_argument("--save-dir", default="experiments/benchmark_results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    builder, weights = MODEL_FACTORY[args.model]
    model = builder(weights=weights).eval().to(device)

    # Static input for model-only timing
    x = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)

    def step():
        with torch.no_grad():
            return model(x)

    # Optional pipeline step: in-the-loop preprocessing + postprocessing on top of inference
    pipeline_step = None
    if args.pipeline:
        # synthetic input that mimics raw input arriving each tick
        raw = torch.randn(3, args.input_size, args.input_size)

        def pipeline_step():
            # "preprocess": normalize + move to device + add batch dim
            inp = ((raw - 0.5) / 0.5).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp)
            # "postprocess": argmax + softmax of top
            _ = out.softmax(dim=1).argmax(dim=1).item()

    n_params = sum(p.numel() for p in model.parameters())
    extras = {
        "model": args.model,
        "input_shape": list(x.shape),
        "params_millions": round(n_params / 1e6, 2),
    }

    report = bench_full(
        step,
        name=f"{args.model}-{args.device}-fp32",
        device=args.device,
        warmup=args.warmup,
        repeat=args.repeat,
        memory_iters=args.memory_iters,
        pipeline_step=pipeline_step,
        pipeline_duration_s=args.pipeline_duration,
        extras=extras,
    )

    print(format_report(report))
    path = save_report_json(report, out_dir=args.save_dir)
    print(f"\nsaved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
