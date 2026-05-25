"""Export a YOLOv8 model to ONNX.

Examples:
    python export_onnx.py --model yolov8n.pt
    python export_onnx.py --model yolov8n.pt --imgsz 416 --opset 17
    python export_onnx.py --model yolov8s.pt --half     # FP16 ONNX
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 PyTorch weights to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--half", action="store_true",
                        help="Export FP16 ONNX (only meaningful when run on FP16-capable HW).")
    parser.add_argument("--dynamic", action="store_true",
                        help="Allow dynamic batch / spatial dims in the ONNX graph.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = YOLO(args.model)
    path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        half=args.half,
        dynamic=args.dynamic,
        simplify=True,
    )
    print(f"wrote: {path}")
    print(f"size : {Path(path).stat().st_size / 1024 / 1024:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
