# Chapter 11 — Edge AI for sensors and time-series

> **Goal:** Extend edge AI beyond camera to industrial sensors, IoT devices, and wearables. By the end of this chapter you should be able to window a sensor stream, train a small model on the windows, deploy an inference loop, and emit alerts on detected anomalies.

Camera AI is the most visible application of Edge AI, but **sensor AI is the larger market** in dollars: predictive maintenance, condition monitoring, asset tracking, environmental sensing, wearables. The pipelines look different from camera pipelines because the input is continuous numeric time-series, not RGB frames.

---

## 1. The pipeline shape

```
[ sensor stream ]
       │ (Hz to kHz)
       v
[ windowing ]         ← sliding window of N samples
       │
       v
[ feature extraction ]   ← stats, FFT, mel-spec, or raw
       │
       v
[ tiny model ]            ← autoencoder, 1D CNN, small RNN, or classifier
       │
       v
[ anomaly / class score ]
       │
       v
[ alert | MQTT | dashboard | log ]
```

Key differences from camera AI:

- **Continuous and asynchronous** — frames have a natural FPS; sensors stream forever.
- **Windowing is half the job.** Picking window size and stride determines what the model can see.
- **Features matter more than depth.** A 1D CNN over raw signal often loses to a shallow model over spectral features.
- **Label scarcity.** You usually have lots of normal data and very few anomalies. Anomaly detection and one-class methods dominate.

---

## 2. Common sensors and signal classes

| Signal | Typical rate | Typical use | Example model |
|---|---|---|---|
| 3-axis accelerometer (IMU) | 50-1000 Hz | Vibration, fall detection, gesture, machine state | 1D CNN, MFCC + small classifier |
| Microphone | 8-48 kHz | Keyword spotting (Ch 12), acoustic anomaly | MFCC / mel-spec + small CNN |
| Temperature / humidity | 0.1-10 Hz | HVAC, cold chain | Statistical + threshold or LSTM |
| Current / voltage | 1-10 kHz | Power monitoring, motor health | FFT + classifier |
| Pressure / flow | 10-100 Hz | Pipe leak, hydraulic anomaly | Autoencoder on windows |
| GPS / position | 1 Hz | Asset tracking, geofence | Rule-based + simple anomaly |

The course's worked example uses **3-axis accelerometer vibration data** with **autoencoder-based anomaly detection** — see `projects/project_03_sensor_anomaly_detection/`.

---

## 3. Windowing

A sliding window converts a continuous stream into fixed-shape inputs the model expects.

```python
def sliding_windows(signal: np.ndarray, win: int, stride: int):
    """signal: (T, channels) → (n_windows, win, channels)"""
    n = (len(signal) - win) // stride + 1
    return np.stack([signal[i*stride : i*stride + win] for i in range(n)])
```

Design choices:

- **Window length** — long enough to contain the slowest pattern you care about (e.g. ≥ 2 cycles of a 10 Hz vibration → ≥ 200 ms at 1 kHz sample rate).
- **Stride** — small stride = overlap, more windows per second, smoother decisions; large stride = independent windows, cheaper.
- **Padding policy** — for real-time, just wait for the next window; don't pad with zeros (it injects an unnatural signal).

For *real-time* inference, you do not pre-compute all windows. You keep a ring buffer and advance it by `stride` samples per inference.

---

## 4. Feature extraction

Whether to extract features depends on the signal and the model:

| Feature | When to use |
|---|---|
| Raw signal | 1D CNN over enough data; convolution learns features |
| Statistics per window (mean, std, kurtosis, peak-to-peak, RMS) | Small sensors, very tight memory, simple classifier sufficient |
| FFT magnitude | Vibration / acoustic anomalies tied to specific frequencies |
| Mel-spectrogram + log | Audio (keyword spotting, anomaly) — close to image, then CNN |
| Hand-engineered features (frequency peaks, spectral entropy, etc.) | When you understand the physics |

For learners new to time-series, **start with statistics or FFT magnitude per window** before reaching for 1D CNNs.

---

## 5. Anomaly detection vs supervised classification

You almost always have lots of "normal" data and very few labeled anomalies. Three approaches:

### 5.1 Autoencoder reconstruction error
- Train an autoencoder to reconstruct normal windows.
- At inference, compute reconstruction error per window.
- High error → anomaly.

Simple, common, and the course's default for `project_03_sensor_anomaly_detection`.

### 5.2 One-class SVM or Isolation Forest
- Train on normal windows only.
- Score = distance from the normal cluster.

Works on hand-engineered features; very small models.

### 5.3 Supervised classifier (if you have labels)
- Standard cross-entropy over class labels.
- Works when you have a balanced dataset of e.g. "healthy" vs "bearing fault" vs "imbalance" vs "misalignment".

If you have labels, use them. If not, autoencoder is the safe default.

---

## 6. MQTT for IoT alerts

When the edge device detects an anomaly, it usually needs to *tell someone*. MQTT is the standard IoT messaging pattern:

```python
import paho.mqtt.client as mqtt
client = mqtt.Client()
client.connect("broker.local", 1883, keepalive=60)

# In your inference loop:
if reconstruction_error > threshold:
    client.publish("factory/line_3/motor_4/anomaly",
                   payload=json.dumps({
                       "ts": time.time(),
                       "device_id": "MOTOR-4",
                       "score": float(reconstruction_error),
                       "window_start": float(window_start_ts),
                   }))
```

For local-only deployments, a simple HTTP POST or a log file is equivalent. The point is: **the edge model's output becomes an event in a larger system**, not just a print.

---

## 7. The worked example: `projects/project_03_sensor_anomaly_detection/`

This project generates a synthetic vibration dataset (so you can run it on any laptop, no hardware needed), trains a tiny autoencoder on the normal regime, and runs an inference loop that flags anomaly events.

Files:

```
project_03_sensor_anomaly_detection/
├── README.md                   — quick start
├── config.yaml                 — sample rate, window, stride, threshold
├── generate_synthetic.py       — make synthetic vibration data
├── train_autoencoder.py        — train the AE on "normal" windows
├── infer_stream.py             — simulate a real-time stream and detect anomalies
└── results/                    — model weights, plots, JSON benchmark
```

Run order:

```bash
cd projects/project_03_sensor_anomaly_detection
python generate_synthetic.py       # → datasets/vibration_normal.npy + vibration_anomaly.npy
python train_autoencoder.py        # → results/autoencoder.pt
python infer_stream.py             # → simulated stream + anomaly flags + plot
```

The README contains the rest.

---

## 8. Edge deployment considerations

For sensor pipelines on real hardware:

- **Microcontrollers** can run the model only if the model is tiny (< 100 KB for TFLite Micro). Chapter 12 covers this.
- **Raspberry Pi / Jetson Nano** can run any of the models in this chapter without issue; the bottleneck is usually I/O (sensor protocol).
- **Sensor sampling at high rates** (e.g. 1 kHz 3-axis IMU) generates a lot of data — keep the windowing buffer in C if you go to MCU.
- **Power consumption** — most edge sensor devices are battery-powered. Inference runs once per window, not per sample.

The course's deliverable is a **simulated stream** that demonstrates the pipeline; deploying to a real MCU is the bonus path through Chapter 12.

---

## 9. Common pitfalls

- **Train/test leakage from overlapping windows.** Always split by time (e.g. first 80% train, last 20% test), not by random sampling — overlapping windows leak nearly-identical samples across the split.
- **Threshold tuned on training data.** Reconstruction error on training data is artificially low. Use a held-out normal set for thresholding.
- **Mixing units across runs.** Calibration drift, sensor swap, firmware change — the model trained on one device may not transfer. Re-validate after any hardware change.
- **Ignoring time-of-day or shift effects.** Many "anomalies" are actually night-shift / weekend regimes. Window features should include a time-of-day signal if relevant.
- **Reporting precision without recall.** If your anomaly rate is 1%, predicting "always normal" yields 99% accuracy. Always report recall on the anomaly class.

---

## 10. What you should be able to do after this chapter

- Window a sensor stream and extract simple features.
- Train an autoencoder on normal data, choose a threshold, and detect anomalies on a held-out set.
- Build a real-time inference loop that emits alerts.
- Decide between autoencoder, one-class, and supervised based on label availability.
- Estimate compute and memory needed to run the model on a given edge device.

---

## 11. Files produced by this chapter

- `docs/11_sensor_ai_timeseries.md` — this file.
- `notebooks/chapter_08_sensor_anomaly_detection.ipynb` — chapter notebook.
- `projects/project_03_sensor_anomaly_detection/` — full mini-project.
