"""Real-time YOLO loop on a video file or webcam.

Mirrors src/inference/camera_loop.py but uses ultralytics YOLO for inference
+ NMS + class names + the annotated plot.

Examples:
    python run_video.py --source 0                              # live webcam
    python run_video.py --source ../../datasets/sample_video.mp4 --no-display --max-frames 100
    python run_video.py --source rtsp://... --record results/cctv.mp4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO real-time camera/video loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", default="0",
                        help="cv2.VideoCapture source: '0' (default webcam), a video path, or RTSP URL.")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fps-window", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = YOLO(args.model)

    cap_arg: int | str = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(cap_arg)
    if not cap.isOpened():
        raise SystemExit(f"failed to open source: {args.source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer: cv2.VideoWriter | None = None
    if args.record is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record.as_posix(), fourcc, 25.0, (width, height))

    recent: list[float] = []
    last_t = time.perf_counter()
    frame_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            results = model.predict(
                source=frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                device=None if args.device == "auto" else args.device,
                verbose=False,
            )
            inf_ms = (time.perf_counter() - t0) * 1000.0
            r = results[0]
            annotated = r.plot()

            now = time.perf_counter()
            recent.append(now - last_t)
            last_t = now
            if len(recent) > args.fps_window:
                recent.pop(0)
            fps = len(recent) / sum(recent) if recent else 0.0

            # FPS overlay (ultralytics already drew boxes + labels)
            h, w = annotated.shape[:2]
            cv2.rectangle(annotated, (0, h - 30), (w, h), (0, 0, 0), -1)
            cv2.putText(annotated, f'{fps:5.1f} FPS  |  {inf_ms:5.1f} ms inf  |  '
                        f'{len(r.boxes)} det',
                        (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

            if writer is not None:
                writer.write(annotated)
            if not args.no_display:
                cv2.imshow("yolo_camera", annotated)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"wrote {args.record}")
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"processed {frame_count} frames; final FPS ~{fps:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
