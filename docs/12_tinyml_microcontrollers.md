# Chapter 12 — TinyML and microcontroller AI

> **Goal:** Understand TinyML — running AI on microcontrollers with KB-scale RAM and Flash. By the end of this chapter you should know what TinyML is, when it is the right choice, and how the train → quantize → deploy workflow differs from the previous chapters.

This chapter is mostly *concept and decision*. Running an actual model on a Cortex-M chip or an ESP32 requires a physical board the course does not assume you own. The chapter notebook walks through a keyword-spotting demo entirely on the laptop, packaged so it would be deployable to a board with `tflite-runtime` or TensorFlow Lite Micro if you had one.

---

## 1. What TinyML is

**TinyML** is a sub-field of Edge AI focused on **always-on, battery-powered, microcontroller-class inference**. The defining constraints:

- **RAM:** 16 KB to 1 MB (typical Cortex-M chip).
- **Flash / model storage:** 256 KB to 16 MB.
- **Compute:** tens to hundreds of MHz, no GPU, sometimes a DSP.
- **Power:** sub-milliwatt average, possibly months on a coin cell.
- **OS:** none, or a tiny RTOS (FreeRTOS, Zephyr).

The result is a different model design space. A 5 MB MobileNet that runs fine on a Pi is much too large for a microcontroller; a 50 KB keyword-spotting model is normal.

---

## 2. Where TinyML wins

TinyML applications share three properties:

1. **Always-on** — must run continuously, often listening for an event.
2. **Privacy-sensitive** — audio, biosignals, that should not leave the device.
3. **Cost-sensitive at scale** — millions of devices, where adding a Pi-class chip is not viable.

Canonical examples:

- **Wake-word spotting** ("Hey Google", "Alexa", "Hey Siri").
- **Gesture recognition** on wearable IMUs (raise-to-wake, swipe).
- **Predictive maintenance** on small vibration sensors with no power budget for a Pi.
- **Animal / bird call detection** in remote, solar-powered devices.
- **Anomaly detection** in industrial sensors at the asset level (Ch 11 + TinyML).

For everything else — anything with a camera, anything that needs > 1 MB RAM, anything not battery-constrained — use a Pi or Jetson and skip the TinyML constraints.

---

## 3. The TinyML toolchain

```
data collection
       ↓
training (TensorFlow / Keras / PyTorch)
       ↓
quantization to INT8 (mandatory; FP32 will not fit)
       ↓
TFLite (.tflite) conversion
       ↓
TFLite Micro (C++) integration  OR  vendor toolchain (X-CUBE-AI, Edge Impulse)
       ↓
flash to MCU  →  measure latency, power, RAM use
```

Two practical paths:

- **TensorFlow Lite Micro (TFLM)** — open-source, C++; the "raw" path.
- **Edge Impulse** — a hosted platform that wraps the same workflow (collect, train, quantize, deploy) into a web UI plus their proprietary `edge-impulse-sdk` runtime. Common in industry.

Vendor-specific: **STMicroelectronics X-CUBE-AI**, **NXP eIQ**, **Renesas e-AI**, **NDP** (Syntiant). They produce smaller binaries than TFLM at the cost of vendor lock-in.

This course defaults to **TFLite Micro** for clarity. The concept transfers to Edge Impulse and vendor SDKs.

---

## 4. Hardware classes

| Board | RAM | Flash | Clock | Typical workloads |
|---|---|---|---|---|
| **Arduino Nano 33 BLE Sense** | 256 KB | 1 MB | 64 MHz | Gestures, simple keyword spotting |
| **ESP32-S3** | 512 KB | 8 MB | 240 MHz | Audio, vision (with PSRAM) |
| **STM32 H7** | 1 MB | 2 MB | 480 MHz | Pro-grade keyword spotting, multi-class |
| **Raspberry Pi Pico** | 264 KB | 2 MB | 133 MHz | Sensor / gesture; tiny models |
| **Syntiant NDP101** | dedicated NN | <100 KB models | always-on | Hardware keyword spotting at <1 mW |
| **Sony Spresense** | 1.5 MB | 8 MB | 156 MHz | Multi-mic audio at low power |

Pick the board *after* you know the model's size and rate.

---

## 5. Model architectures for TinyML

The starting points for the field:

- **DS-CNN (Depthwise Separable CNN)** — Google's keyword-spotting reference; ~30-100 KB.
- **TinyConv / 1D-CNN** — sensor classification; <100 KB.
- **TinyTransformer** — recent (2024+); often still > MCU-class without serious pruning.

The trick is rarely a fancy architecture; it is **aggressive INT8 quantization + a small student** (Ch 8). Distillation from a bigger teacher is common.

---

## 6. The keyword-spotting demo

`projects/project_05_tinyml_keyword_spotting/` packages a small workflow:

1. Generate (or download) a small audio dataset of a few keywords.
2. Compute MFCC features per window.
3. Train a small CNN (~50-100 KB once quantized).
4. Convert to TFLite INT8.
5. Print the would-be flash and RAM footprint.

If you have a board, the same `.tflite` deploys via `tflite-runtime` (on a Pi) or via TFLite Micro (on a Cortex-M, after wrapping the `.tflite` bytes into a C array with `xxd -i`).

If you do not, the workflow still demonstrates the toolchain.

---

## 7. Memory and latency math

TinyML demands you do arithmetic before you train:

- **Model size on flash** = weight bytes + small overhead. INT8 = 1 byte per weight. A 50K-parameter model = ~50 KB on flash.
- **Activation tensor RAM** = largest activation tensor across the forward pass. Often >> model size. A 32×32 conv layer with 32 channels = 32 KB just for that one tensor.
- **Inference latency** = depends on the chip. A 50 KB DS-CNN on a Cortex-M4 at 64 MHz might take 30-100 ms per inference. On an STM32-H7 at 480 MHz with CMSIS-NN INT8, 5-15 ms.
- **Power per inference** = (current at full clock) × (inference time). 30 ms × 15 mA × 3.3 V ≈ 1.5 mJ — for an always-on system at 1 inference per 100 ms, that is ~15 mW continuous.

**Spend more time on the memory budget than on the model architecture.** The architecture is usually obvious once the budget is fixed.

---

## 8. What this chapter does NOT do

- **Deploy to a real board** — assumes none is available.
- **Cover Edge Impulse end-to-end** — its UI changes faster than this doc would; their tutorials are linked instead.
- **Train from scratch on a large audio dataset** — the demo uses a tiny subset for speed.

For learners who have a board, the suggested next steps are:

1. Replace the synthetic / tiny dataset with **Google Speech Commands** (or your own recorded set).
2. Follow Edge Impulse's docs for the same architecture and compare results.
3. Deploy and measure real power.

---

## 9. What you should be able to do after this chapter

- Tell when TinyML is the right fit (and when it is not).
- Sketch the train → quantize → TFLite → MCU workflow.
- Estimate model size and activation RAM for a candidate architecture.
- Read the keyword-spotting demo and adapt it to a different small classification task.
- Decide between TFLite Micro and Edge Impulse for a hypothetical product.

---

## 10. Files produced by this chapter

- `docs/12_tinyml_microcontrollers.md` — this file.
- `notebooks/chapter_09_tinyml_intro.ipynb` — concept walkthrough with a small audio classifier.
- `projects/project_05_tinyml_keyword_spotting/` — keyword-spotting demo (concept + workflow).
