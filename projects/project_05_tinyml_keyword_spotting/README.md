# Project 05 — TinyML keyword spotting (concept demo)

Worked example for Chapter 12. Trains a small audio classifier on a tiny synthetic dataset, computes the memory footprint a microcontroller would need, and demonstrates the workflow you would deploy to an MCU.

The project runs on a laptop with **no microcontroller required**. If you do have a board (Arduino Nano 33 BLE Sense, ESP32-S3, STM32-H7, etc.), the produced `.tflite` file is the same one you would flash via TFLite Micro or via Edge Impulse.

---

## Why a *concept* demo

Real TinyML keyword spotting takes hours of data and a board to validate. The course goal here is to show:

- The pipeline shape (MFCC → small CNN → INT8 quantize → estimate footprint).
- The memory math (does this model fit in 256 KB RAM?).
- The accuracy / size trade-off.

For real deployment, follow this demo's structure but swap the dataset for **Google Speech Commands** (or your own), and the runtime for **TensorFlow Lite Micro** on a real chip.

---

## Files

```
project_05_tinyml_keyword_spotting/
├── README.md                  — this file
├── config.yaml                — sample rate, MFCC params, model size
├── generate_synthetic.py      — tiny synthetic audio dataset (3 classes)
├── train_kws.py               — train a small DS-CNN classifier
├── footprint.py               — estimate model + activation memory
└── results/                   — model, plots, footprint estimate
```

---

## Quick start

```bash
cd projects/project_05_tinyml_keyword_spotting

# 1. Generate the synthetic 3-class dataset
python generate_synthetic.py
# → ../../datasets/kws_synthetic.npz

# 2. Train the model
python train_kws.py
# → results/kws_model.pt
# → results/training_curves.png

# 3. Estimate memory footprint
python footprint.py
# → printed memory budget table
```

The dataset is synthetic on purpose — three distinguishable spectral patterns labeled "up", "down", "stop". A real keyword-spotting model would use Google Speech Commands or similar.

---

## Configuration

`config.yaml`:

```yaml
sample_rate_hz: 16000
window_ms: 30
hop_ms: 10
n_mfcc: 10
n_classes: 3              # synthetic dataset has 3 classes
samples_per_class: 200
epochs: 25
batch_size: 32
seed: 0
```

For a real Speech Commands run, change to `n_classes: 10` (or more), and replace `generate_synthetic.py` with a loader for the Speech Commands `.wav` files.

---

## What the model looks like

A tiny depthwise-separable CNN:

- **Input:** 49 × 10 MFCC frames (1 second of audio at 100 fps, 10 coefficients per frame).
- **Conv stack:** depthwise + pointwise convs at small channel counts.
- **GAP + linear classifier.**
- **Output:** logits over `n_classes` keywords.

Quantized to INT8, this should fit well under **100 KB on flash** and under **64 KB activation RAM**. The `footprint.py` script does the math from the saved model.

---

## Deploying to a real microcontroller (overview)

If you have a Cortex-M board:

1. **Convert** the trained model to TFLite INT8:
   ```python
   import tensorflow as tf
   converter = tf.lite.TFLiteConverter.from_saved_model("path/to/savedmodel")
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   converter.representative_dataset = representative_generator  # ~100 real samples
   converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
   converter.inference_input_type = tf.int8
   converter.inference_output_type = tf.int8
   tflite_bytes = converter.convert()
   open("kws.tflite", "wb").write(tflite_bytes)
   ```

2. **Embed** the `.tflite` bytes as a C array:
   ```bash
   xxd -i kws.tflite > kws_model.cc
   ```

3. **Link** with **TensorFlow Lite Micro** in your firmware project (Arduino IDE, PlatformIO, STM32CubeIDE — all have TFLM examples).

4. **Measure** real inference time, RAM use, and power.

The `.tflite` file is what travels between this demo and a real MCU.

**Edge Impulse alternative:** instead of TFLM, you can upload the same data to Edge Impulse, train via their UI, and download a ready-to-flash library for your board. The workflow is the same; only the toolchain differs.

---

## What to put in the report

| Field | Example |
|---|---|
| Target hardware (real or hypothetical) | Arduino Nano 33 BLE Sense (Cortex-M4 @ 64 MHz, 256 KB RAM, 1 MB Flash) |
| Sample rate / window / hop | 16 kHz / 30 ms / 10 ms |
| Number of classes | 3 (synthetic) |
| Model size (INT8) | 28 KB |
| Activation RAM peak | 11 KB |
| Inference latency (laptop, ORT INT8) | 0.8 ms |
| Inference latency (MCU, estimate) | ~30 ms |
| Test accuracy | 0.96 |
