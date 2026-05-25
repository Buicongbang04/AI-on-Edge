"""Simulate a real-time inference loop on the anomaly-injected signal (Chapter 11).

Reads `datasets/vibration_anomaly.npy`, iterates over it in windows (with the
same window/stride as training), computes reconstruction error per window, and
emits an alert when ≥ `alert_min_consecutive` consecutive windows exceed the
threshold (suppresses single-window spikes).

Outputs:
    results/stream_with_anomalies.png  — plot of x-axis signal with flagged spans
    results/anomaly_events.json        — alerts in production-style JSON
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from train_autoencoder import TinyAutoencoder, windowize


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--data", type=Path,
                        default=ROOT.parent.parent / "datasets" / "vibration_anomaly.npy")
    parser.add_argument("--device-id", default="MOTOR-SYNTH")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    win = cfg["window_samples"]
    stride = cfg["stride_samples"]
    rate = cfg["sample_rate_hz"]

    # Load model + threshold
    ckpt_path = ROOT / "results" / "autoencoder.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"missing {ckpt_path}; run train_autoencoder.py first")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TinyAutoencoder(in_dim=ckpt["in_dim"], latent_dim=cfg["latent_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    threshold = float(ckpt["threshold"])
    print(f"loaded model from {ckpt_path}  threshold={threshold:.5f}")

    if not args.data.exists():
        raise SystemExit(f"missing {args.data}; run generate_synthetic.py first")
    signal = np.load(args.data)
    print(f"loaded stream: {signal.shape}  ({signal.shape[0] / rate:.2f} s)")

    windows = windowize(signal, win, stride)
    timestamps = np.arange(len(windows)) * stride / rate  # seconds (window start)
    print(f"will process {len(windows)} windows")

    # Inference loop (with simulated wall-clock timing)
    inf_times_ms: list[float] = []
    scores = np.empty(len(windows), dtype=np.float32)
    with torch.no_grad():
        for i, w in enumerate(windows):
            x = torch.from_numpy(w[None, :])
            t0 = time.perf_counter()
            recon = model(x)
            err = float(((x - recon) ** 2).mean().item())
            inf_times_ms.append((time.perf_counter() - t0) * 1000.0)
            scores[i] = err

    inf_times_ms = np.array(inf_times_ms)
    print(f"\ninference latency: mean={inf_times_ms.mean():.3f} ms  "
          f"P95={np.percentile(inf_times_ms, 95):.3f} ms  "
          f"throughput={len(windows) / (inf_times_ms.sum() / 1000.0):.0f} windows/s")

    # Group consecutive over-threshold windows into events
    over = scores > threshold
    events = []
    i = 0
    while i < len(over):
        if over[i]:
            j = i
            while j < len(over) and over[j]:
                j += 1
            run_len = j - i
            if run_len >= cfg["alert_min_consecutive"]:
                event = {
                    "ts": float(time.time()),
                    "device_id": args.device_id,
                    "window_start_s": float(timestamps[i]),
                    "window_end_s": float(timestamps[j - 1] + win / rate),
                    "score_max": float(scores[i:j].max()),
                    "score_mean": float(scores[i:j].mean()),
                    "threshold": threshold,
                    "consecutive_windows": int(run_len),
                }
                events.append(event)
            i = j
        else:
            i += 1
    print(f"detected {len(events)} anomaly events")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot the x-axis signal with flagged spans
    plt.figure(figsize=(12, 3.5))
    t_full = np.arange(len(signal)) / rate
    plt.plot(t_full, signal[:, 0], linewidth=0.5, color="#1565C0", label="acc_x")
    for ev in events:
        plt.axvspan(ev["window_start_s"], ev["window_end_s"], color="red", alpha=0.25,
                    label="_anomaly")
    if events:
        plt.plot([], [], color="red", alpha=0.5, linewidth=8, label="anomaly span")
    plt.xlabel("time (s)")
    plt.ylabel("acc_x")
    plt.title(f"Stream with {len(events)} anomaly events detected")
    plt.legend(loc="upper right")
    plt.tight_layout()
    png = out_dir / "stream_with_anomalies.png"
    plt.savefig(png, dpi=120)
    print(f"wrote {png}")

    # Events JSON
    js = out_dir / "anomaly_events.json"
    js.write_text(json.dumps(events, indent=2))
    print(f"wrote {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
