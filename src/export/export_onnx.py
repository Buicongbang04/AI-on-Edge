"""Export a PyTorch torchvision classifier to ONNX (Chapter 5).

Produces:
    experiments/exported_models/<model>.onnx       — the ONNX file
    plus optional output-comparison vs PyTorch to confirm the export is correct.

Example:
    python src/export/export_onnx.py --model mobilenet_v3_small
    python src/export/export_onnx.py --model resnet50 --opset 17 --dynamic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import models


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


def export_model(
    name: str,
    out_path: Path,
    *,
    input_size: int = 224,
    opset: int = 17,
    dynamic: bool = False,
) -> Path:
    """Export a torchvision classifier to ONNX. Returns the output path."""
    if name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(MODEL_FACTORY)}")

    builder, weights = MODEL_FACTORY[name]
    model = builder(weights=weights).eval()

    dummy = torch.randn(1, 3, input_size, input_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Dynamic axes — allow variable batch (and optionally H, W) at runtime.
    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch"},
        }

    # Use the legacy (TorchScript-based) exporter via dynamo=False. The newer dynamo
    # exporter splits weights into a separate `.onnx.data` file, which is fine for
    # production but confusing for a teaching context — students expect one file.
    torch.onnx.export(
        model,
        dummy,
        out_path.as_posix(),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    return out_path


def validate_onnx(path: Path) -> None:
    """Run onnx.checker on the exported file."""
    import onnx
    model = onnx.load(path.as_posix())
    onnx.checker.check_model(model)
    print(f"  ONNX check OK. ir_version={model.ir_version} opset={model.opset_import[0].version}")


def compare_outputs(
    name: str,
    onnx_path: Path,
    *,
    input_size: int = 224,
    tol: float = 1e-4,
) -> dict[str, float]:
    """Compare PyTorch and ONNX Runtime outputs on the same random input.

    Returns max abs diff and cosine similarity between flattened outputs.
    """
    import onnxruntime as ort

    builder, weights = MODEL_FACTORY[name]
    model = builder(weights=weights).eval()
    x = torch.randn(1, 3, input_size, input_size)

    with torch.no_grad():
        y_torch = model(x).numpy()

    sess = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    y_ort = sess.run(None, {"input": x.numpy()})[0]

    max_abs_diff = float(np.max(np.abs(y_torch - y_ort)))
    a, b = y_torch.flatten(), y_ort.flatten()
    cosine = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    same_argmax = int(np.argmax(y_torch, axis=1)[0] == np.argmax(y_ort, axis=1)[0])
    ok = max_abs_diff < tol or cosine > 0.999

    print(f"  max|diff|={max_abs_diff:.6f}  cosine={cosine:.6f}  same_argmax={same_argmax}  "
          f"{'OK' if ok else 'WARNING'}")
    return {
        "max_abs_diff": max_abs_diff,
        "cosine_similarity": cosine,
        "same_argmax": bool(same_argmax),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a PyTorch classifier to ONNX and validate it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODEL_FACTORY), default="mobilenet_v3_small")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/exported_models"))
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true",
                        help="Export with dynamic batch and spatial dims.")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip onnx.checker validation.")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip PyTorch-vs-ONNX output comparison.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out_dir / f"{args.model}{'-dyn' if args.dynamic else ''}.onnx"

    print(f"exporting {args.model} → {out}")
    export_model(args.model, out, input_size=args.input_size, opset=args.opset, dynamic=args.dynamic)
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"  saved {out}  size={size_mb:.2f} MB")

    if not args.no_validate:
        validate_onnx(out)
    if not args.no_compare:
        compare_outputs(args.model, out, input_size=args.input_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
