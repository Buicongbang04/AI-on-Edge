"""Real-time camera inference loop (Chapter 9).

Captures frames from a webcam (or a video file), runs an ONNX classifier on each
frame, draws the top-1 prediction + FPS on the frame, and displays it via OpenCV.

Optional:
    --no-display   : disable OpenCV window (e.g. when running over SSH without X)
    --record path  : save the annotated video to a file
    --skip N       : skip every N frames between inferences (use last prediction)
    --max-frames N : stop after N frames

Examples:
    python src/inference/camera_loop.py \
        --model experiments/exported_models/mobilenet_v3_small.onnx
    python src/inference/camera_loop.py \
        --model experiments/exported_models/mobilenet_v3_small.onnx \
        --source 0 --record outputs/camera_demo.mp4

Notes:
    - On headless machines (no display server), pass --no-display and use --max-frames.
    - For real edge deployment, use a CSI camera on Jetson / Raspberry Pi for much lower
      capture latency than a USB webcam.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from torchvision.models import MobileNet_V3_Small_Weights


_PREP = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
_CLASSES = MobileNet_V3_Small_Weights.IMAGENET1K_V1.meta["categories"]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def preprocess_frame(bgr_frame: np.ndarray) -> np.ndarray:
    """OpenCV BGR frame → ImageNet-preprocessed tensor (1, 3, 224, 224)."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    # PIL-style; torchvision transforms expect a PIL Image, but they also accept tensor.
    # To avoid PIL dependency on hot path, do it inline:
    h, w = rgb.shape[:2]
    # Resize so the shorter side is 232, then center-crop to 224
    target = 232
    if h < w:
        new_h, new_w = target, int(w * target / h)
    else:
        new_h, new_w = int(h * target / w), target
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    y0 = (new_h - 224) // 2
    x0 = (new_w - 224) // 2
    crop = resized[y0:y0 + 224, x0:x0 + 224]
    # To CHW float32 in 0..1, then normalize with ImageNet stats
    chw = crop.astype(np.float32) / 255.0
    chw = chw.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    chw = (chw - mean) / std
    return chw[None, ...]   # add batch dim


def draw_overlay(frame: np.ndarray, label: str, fps: float, latency_ms: float) -> None:
    """Mutate `frame` in place to add label / FPS / latency overlay."""
    h, w = frame.shape[:2]
    # background banner at the top
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'{fps:5.1f} FPS  |  {latency_ms:5.1f} ms / inference',
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time camera classifier (Chapter 9).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to ONNX classifier (e.g. mobilenet_v3_small.onnx).")
    parser.add_argument("--source", default="0",
                        help="cv2.VideoCapture source: '0' for default webcam, "
                             "or a path to a video file, or an RTSP URL.")
    parser.add_argument("--width", type=int, default=640,
                        help="Capture width (USB webcams may downsample silently).")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--skip", type=int, default=0,
                        help="Run inference every (skip+1) frames; keep last prediction otherwise.")
    parser.add_argument("--no-display", action="store_true",
                        help="Do not open an OpenCV window (useful headless).")
    parser.add_argument("--record", type=Path, default=None,
                        help="Save annotated frames to this video path.")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after this many frames (0 = no limit).")
    parser.add_argument("--fps-window", type=int, default=30,
                        help="Number of recent frames used to compute the displayed FPS.")
    return parser.parse_args()


def open_source(source: str, width: int, height: int) -> cv2.VideoCapture:
    cap_arg: int | str = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_arg)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f"failed to open video source: {source}")
    return cap


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")

    print(f"loading {args.model} via {args.provider}")
    sess = ort.InferenceSession(args.model.as_posix(), providers=[args.provider])
    in_name = sess.get_inputs()[0].name

    cap = open_source(args.source, args.width, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video source open: {actual_w}x{actual_h}")

    writer: cv2.VideoWriter | None = None
    if args.record is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record.as_posix(), fourcc, 25.0, (actual_w, actual_h))

    label = "..."
    inf_ms = 0.0
    recent_iter_times: list[float] = []
    frame_count = 0
    last_iter = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("source closed; exiting")
                break

            run_inference_this_frame = (frame_count % (args.skip + 1) == 0)
            if run_inference_this_frame:
                x = preprocess_frame(frame)
                t0 = time.perf_counter()
                (logits,) = sess.run(None, {in_name: x})
                inf_ms = (time.perf_counter() - t0) * 1000.0
                probs = softmax(logits, axis=-1)[0]
                idx = int(probs.argmax())
                label = f'{_CLASSES[idx]} ({probs[idx] * 100:.1f}%)'

            now = time.perf_counter()
            recent_iter_times.append(now - last_iter)
            last_iter = now
            if len(recent_iter_times) > args.fps_window:
                recent_iter_times.pop(0)
            fps = len(recent_iter_times) / sum(recent_iter_times) if recent_iter_times else 0.0

            draw_overlay(frame, label, fps, inf_ms)

            if writer is not None:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("camera_loop", frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):  # q or ESC
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
