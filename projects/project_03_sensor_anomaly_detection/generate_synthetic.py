"""Generate a synthetic 3-axis accelerometer dataset for Chapter 11.

Normal regime: sinusoid at 12 Hz (a healthy motor) + small white noise.
Anomalous regime: same baseline + occasional bursts at 47 Hz (imbalance fault).

Writes:
    datasets/vibration_normal.npy    shape (T, 3), float32
    datasets/vibration_anomaly.npy   shape (T, 3), float32
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def synth_normal(seconds: float, rate_hz: int = 1000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(seconds * rate_hz), endpoint=False, dtype=np.float32)
    base = np.stack([
        0.50 * np.sin(2 * np.pi * 12.0 * t),
        0.40 * np.sin(2 * np.pi * 12.0 * t + 0.3),
        0.45 * np.sin(2 * np.pi * 12.0 * t + 0.7),
    ], axis=1)
    noise = 0.05 * rng.standard_normal(base.shape).astype(np.float32)
    return (base + noise).astype(np.float32)


def synth_anomalous(seconds: float, rate_hz: int = 1000, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal = synth_normal(seconds, rate_hz, seed=seed + 100)
    t = np.linspace(0, seconds, int(seconds * rate_hz), endpoint=False, dtype=np.float32)
    # Inject 6 bursts of imbalance fault at random starts
    n_bursts = 6
    burst_len_samples = int(0.3 * rate_hz)
    for _ in range(n_bursts):
        start = rng.integers(0, len(t) - burst_len_samples)
        burst_t = t[start:start + burst_len_samples] - t[start]
        burst = np.stack([
            0.35 * np.sin(2 * np.pi * 47.0 * burst_t),
            0.30 * np.sin(2 * np.pi * 47.0 * burst_t + 0.1),
            0.32 * np.sin(2 * np.pi * 47.0 * burst_t + 0.4),
        ], axis=1)
        signal[start:start + burst_len_samples] += burst.astype(np.float32)
    return signal


def main() -> int:
    out_dir = Path(__file__).resolve().parents[2] / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    normal = synth_normal(seconds=20.0)
    anomaly = synth_anomalous(seconds=10.0)
    np.save(out_dir / "vibration_normal.npy", normal)
    np.save(out_dir / "vibration_anomaly.npy", anomaly)
    print(f"wrote {out_dir / 'vibration_normal.npy'}  shape={normal.shape}")
    print(f"wrote {out_dir / 'vibration_anomaly.npy'} shape={anomaly.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
