"""Generate a tiny synthetic 'audio' dataset for the TinyML KWS demo.

Three classes:
- "up":    a chirp from 200 Hz → 1500 Hz
- "down":  a chirp from 1500 Hz → 200 Hz
- "stop":  two-tone burst (600 Hz, 900 Hz)

Each sample is 1 s long at 16 kHz with small added noise.

This is NOT a real keyword-spotting dataset. It exists so the demo runs in seconds
without any audio download. For real KWS, swap this for Google Speech Commands.

Writes:  datasets/kws_synthetic.npz   {"X": (N, 16000), "y": (N,), "names": [...]}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def chirp(t: np.ndarray, f0: float, f1: float) -> np.ndarray:
    # Linear chirp
    return np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * t[-1]) * t * t))


def make_sample(label: int, rng: np.random.Generator, rate: int = 16000) -> np.ndarray:
    t = np.linspace(0, 1.0, rate, endpoint=False, dtype=np.float32)
    if label == 0:           # up
        s = chirp(t, 200, 1500)
    elif label == 1:         # down
        s = chirp(t, 1500, 200)
    else:                    # stop
        s = 0.6 * np.sin(2 * np.pi * 600 * t) + 0.4 * np.sin(2 * np.pi * 900 * t)
    s = s.astype(np.float32)
    # Random amplitude + small noise so samples are not identical
    amp = 0.5 + 0.5 * rng.random()
    return (amp * s + 0.02 * rng.standard_normal(rate).astype(np.float32)).astype(np.float32)


def main() -> int:
    rng = np.random.default_rng(0)
    n_per_class = 200
    rate = 16000
    classes = ["up", "down", "stop"]

    X, y = [], []
    for label in range(len(classes)):
        for _ in range(n_per_class):
            X.append(make_sample(label, rng, rate))
            y.append(label)
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)
    print("dataset:", X.shape, y.shape)

    out_dir = Path(__file__).resolve().parents[2] / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kws_synthetic.npz"
    np.savez(out_path, X=X, y=y, names=np.array(classes))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
