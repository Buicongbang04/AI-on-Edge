"""Run YOLOv8 on a single image and save the annotated output."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO single-image inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", default="yolov8n.pt",
                        help="ultralytics-supported model: pt / onnx / engine / openvino path.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.exists():
        raise SystemExit(f"image not found: {args.image}")
    args.out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    t0 = time.perf_counter()
    results = model.predict(
        source=args.image.as_posix(),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=None if args.device == "auto" else args.device,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    result = results[0]
    annotated = result.plot()
    out_path = args.out / f"{args.image.stem}_annotated.jpg"
    cv2.imwrite(out_path.as_posix(), annotated)

    print(f"detections: {len(result.boxes)}")
    for b in result.boxes:
        cls_id = int(b.cls.item())
        cls_name = result.names[cls_id]
        score = float(b.conf.item())
        xyxy = [round(x, 1) for x in b.xyxy.tolist()[0]]
        print(f"  {cls_name:<15s} score={score:.2f}  box={xyxy}")
    print(f"\nlatency: {elapsed_ms:.1f} ms (single image, includes load + IO)")
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
