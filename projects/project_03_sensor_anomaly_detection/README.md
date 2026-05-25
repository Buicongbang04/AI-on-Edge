# Project 03 — Sensor anomaly detection

Worked example for Chapter 11. Trains a tiny autoencoder on synthetic vibration data, then runs a simulated real-time inference loop that flags anomaly windows.

You do **not** need any sensor hardware. The whole project runs on a laptop CPU. It is designed to drop in to a Raspberry Pi or Jetson with no code changes — only the data source changes (from `.npy` to a real sensor read).

---

## Files

```
project_03_sensor_anomaly_detection/
├── README.md                  — this file
├── config.yaml                — sample rate, window size, threshold
├── generate_synthetic.py      — make synthetic accelerometer data
├── train_autoencoder.py       — train a small AE on normal windows
├── infer_stream.py            — simulate a stream and flag anomalies
└── results/                   — model, plots, JSON
```

---

## Quick start

```bash
cd projects/project_03_sensor_anomaly_detection

# 1. Generate synthetic vibration data (normal + anomalous)
python generate_synthetic.py
# → ../../datasets/vibration_normal.npy   (~20 s of 'healthy' vibration)
# → ../../datasets/vibration_anomaly.npy  (~10 s with injected anomalies)

# 2. Train autoencoder on normal data only
python train_autoencoder.py
# → results/autoencoder.pt
# → results/training_loss.png

# 3. Simulate a stream and detect anomalies
python infer_stream.py
# → results/stream_with_anomalies.png
# → results/anomaly_events.json
```

---

## What the synthetic data looks like

`generate_synthetic.py` produces a 3-axis accelerometer signal at 1 kHz:

- **Normal regime:** sinusoidal vibration at 12 Hz (a healthy motor) + low-amplitude noise.
- **Anomaly regime:** occasional bursts at 47 Hz (an imbalance fault) added on top of the normal signal.

This is a *toy* dataset. Real machine vibration is messier, but the pipeline is the same.

---

## What the autoencoder does

- Input: a window of length 128 samples × 3 axes (= 128 ms at 1 kHz, flattened to 384 dims).
- Encoder: small MLP that compresses to a 16-d latent.
- Decoder: mirror MLP that reconstructs the window.
- Loss: mean squared reconstruction error.

Anomaly score per window = reconstruction MSE. We pick the threshold as the 99th percentile of reconstruction error on a held-out normal set.

---

## Configuration

`config.yaml`:

```yaml
sample_rate_hz: 1000
window_samples: 128
stride_samples: 64
n_channels: 3
latent_dim: 16
epochs: 30
batch_size: 64
threshold_quantile: 0.99   # of reconstruction error on held-out normal
alert_min_consecutive: 2   # require K consecutive flagged windows before emitting an alert
```

Knobs you will want to tune for real data:

- `window_samples` — long enough to contain ≥2 cycles of the lowest fault frequency.
- `threshold_quantile` — lower = more sensitive (more false alarms); higher = fewer alerts.
- `alert_min_consecutive` — suppresses single-window spikes; raises the bar for issuing an alert.

---

## Reading the output

After `infer_stream.py`, two artifacts in `results/`:

- `stream_with_anomalies.png` — the time-series with flagged anomaly windows highlighted.
- `anomaly_events.json` — list of `{ ts, score, window_start }` records, the kind of payload you would publish over MQTT in production.

A typical event:

```json
{
  "ts": 1737721234.56,
  "device_id": "MOTOR-SYNTH",
  "window_start_s": 12.288,
  "window_end_s":   12.416,
  "score": 0.0481,
  "threshold": 0.0123
}
```

---

## Adapting to real sensor hardware

The only change is the **data source** in `infer_stream.py`. Replace the file-backed iterator with a generator that reads from your sensor (I²C IMU, USB serial, SPI ADC, or MQTT input topic). The rest of the pipeline — windowing, model, alerting — is identical.

For an IMU sample (Bosch BMI270, ICM-20948):

```python
import smbus2
bus = smbus2.SMBus(1)
def read_imu():
    while True:
        # read 6 bytes (3 axes × 2 bytes), convert to g, yield (ax, ay, az)
        ...
```

For an MQTT-fed sensor:

```python
import paho.mqtt.client as mqtt
def read_mqtt_stream(topic):
    # yields samples as they arrive on the topic
    ...
```

The notebook walks through the whole pipeline interactively.

---

## Reporting

Drop your benchmark + anomaly precision/recall numbers into `results/benchmark.json`. Format suggestion:

| Field | Example |
|---|---|
| Hardware | "Laptop CPU, single core" |
| Window/stride | "128 / 64 samples" |
| Latent dim | 16 |
| Inference latency P95 | "0.4 ms / window" |
| End-to-end FPS | "Windows per second: 2000" |
| Precision (anomaly) | 0.92 |
| Recall (anomaly) | 0.96 |
| F1 (anomaly) | 0.94 |
