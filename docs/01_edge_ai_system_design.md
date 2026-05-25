# Chapter 1 — Edge AI system design

> **Goal:** Frame an Edge AI problem as a *system*, not a model choice. By the end of this chapter you should be able to write a one-page system-design note for any Edge AI application that specifies its input, output, metric, latency budget, hardware, and the failure modes you will guard against.

The single biggest mistake newcomers make when moving from ML to Edge AI is starting from the model: "Which YOLO version should I use?", "Should I use a ResNet or a MobileNet?". This is the wrong starting point. The starting point is **the system**: what comes in, what must go out, how fast, with what budget, on what device, and what happens when something breaks.

This chapter teaches that starting point.

---

## 1. The system view of an Edge AI application

Every Edge AI application can be drawn as the same skeleton:

```
[ input source ] → [ preprocessing ] → [ model inference ] → [ postprocessing ] → [ output / action ]
                                  ↓                                          ↑
                       [ hardware + runtime ]                       [ logging + safety ]
```

Before you pick a model, you fix the six fields around the model:

| Field | Question to answer |
|---|---|
| Input source | What sensor produces the data? At what rate? In what format? |
| Output | What does the system have to emit? A label, a box, a mask, a command, an alert? |
| Metric | How will you decide the system is "good enough"? Not just accuracy — also latency, FPS, recall on the rare class, false-alarm cost |
| Latency budget | What is the maximum acceptable end-to-end response time, P95? |
| Hardware + runtime | What device, OS, and inference engine? What power and thermal envelope? |
| Failure modes | What can break: sensor, model, data drift, network, power? What is the fallback? |

These six fields are the **specification** of the system. The model is one variable you tune to satisfy the spec.

---

## 2. Input source

The first design question is what the sensor actually delivers and at what rate. The right answer determines everything downstream.

**For cameras and video streams:**
- Resolution (e.g. 1920×1080, 4K).
- Color space (RGB, YUV, grayscale, IR).
- Frame rate (15, 30, 60 FPS).
- Compression (raw, MJPEG, H.264 — and where it is decoded).
- Connection (USB, MIPI CSI, GigE, RTSP).
- Lighting conditions (controlled lab, factory floor, outdoor, night).

**For audio:**
- Sample rate (8 kHz, 16 kHz, 44.1 kHz).
- Channels (mono, stereo, array).
- Bit depth.
- Background noise profile.

**For other sensors (IMU, vibration, temperature, current):**
- Sample rate (Hz to kHz).
- Number of axes / channels.
- Quantization (bits).
- Synchronization to other sensors.

**For video / sensor streams:** also specify the **buffering policy.** Do you process every frame, every N-th frame, or whatever arrives between completions? Real-time camera pipelines almost always drop frames; design for it explicitly.

---

## 3. Output

The output of the system is *not* the model's output — it is the side effect the system has on the world.

| Output type | Typical use case | What the system must produce |
|---|---|---|
| Label | Image classifier, sound classifier | One label + confidence per input |
| Bounding boxes | Object detection, people counting | List of (class, box, score) with NMS applied |
| Mask | Defect segmentation, scene parsing | A pixel mask, possibly with class IDs |
| Command | Robot control, valve control, switch | A discrete or continuous action passed to a controller (Ch 14-15) |
| Alert / event | Anomaly detection, safety monitoring | A timestamped event sent to MQTT, log, or dashboard |
| Decision + reasoning | Edge LLM assistant (Ch 18) | A short structured response, often gated by rules |

Many edge systems combine several output types — a YOLO detection plus an alert when a person enters a restricted zone, for example. **Each output type also has its own latency budget**; alerts to a dashboard can tolerate seconds, motor commands cannot.

---

## 4. Metric: not just accuracy

Edge AI metrics fall into three groups. Pick at least one from each, and write down the target.

### 4.1 Model-quality metrics

- Classification: accuracy, top-k accuracy, per-class precision/recall, F1, confusion-matrix structure.
- Detection: mAP, recall at fixed precision, IoU threshold sensitivity.
- Segmentation: mIoU, pixel accuracy.
- Time series / sensor: precision/recall at the chosen confidence threshold, lead time before an event.

Often the meaningful target is *recall on the rare class* (defective products, faults, abnormal events) rather than overall accuracy.

### 4.2 Performance metrics (the edge-specific group)

- `model_size_mb` — size of the deployable artifact.
- `latency_mean_ms` and `latency_p50_ms` — typical latency.
- `latency_p95_ms` and `latency_p99_ms` — tail latency, where failures live.
- `fps_end_to_end` — for camera pipelines, frames per second the whole pipeline can sustain.
- `memory_peak_mb` — RSS / VRAM peak under steady load.
- `cpu_gpu_npu_utilization` — how loaded the accelerator is.
- `temperature_celsius` and `power_watt` — for thermally constrained / battery-powered devices.

Chapter 4 builds the benchmark template that produces these numbers.

### 4.3 Cost metrics (often skipped, often deciding)

- Cost per device.
- Cost per inference (electricity, network, maintenance).
- Cost of a false positive vs cost of a false negative — in money, time, safety, or reputation. *This single ratio often decides which model you ship.*

A defect detector that flags 5% false positives may be unacceptable on a high-volume line; one that misses 5% of defects may be unacceptable for safety-critical parts. The metric is not "accuracy" — it is "value lost per hour from each error type."

---

## 5. Latency budget

Latency budgeting is the discipline that separates a working demo from a deployable system. The budget is a **target end-to-end P95 latency** that you allocate across the pipeline before you build it.

Example budget for a real-time camera classifier at 30 FPS:

```
Target end-to-end P95: 33 ms (one frame at 30 FPS)

Capture (camera + decode)            5 ms
Preprocess (resize + normalize)      3 ms
Model inference                     15 ms
Postprocess (argmax + overlay)       3 ms
Output (display or alert)            5 ms
Slack                                2 ms
---------------------------------- ----
Total                               33 ms
```

If model inference comes in at 25 ms, you have already blown the budget. The decision points are:

- Drop the input resolution (cheap; accept some accuracy loss).
- Skip every other frame (cheap; halves effective FPS).
- Quantize the model to INT8 or FP16 (Ch 8).
- Pick a smaller architecture (most impactful).
- Move from CPU to GPU/NPU (hardware change).
- Add asynchronous capture + inference (threading, Ch 9).

Always write down the budget *before* you start optimizing. Otherwise you optimize random things.

### 5.1 Typical budgets in this course

| Use case | End-to-end P95 budget | Notes |
|---|---|---|
| Real-time camera classifier (30 FPS) | 33 ms | Tight; almost always needs preprocessing optimization |
| Real-time YOLO detector (30 FPS) | 33 ms | Often needs FP16/INT8 and lightweight model |
| Real-time YOLO detector (15 FPS) | 66 ms | More forgiving; useful for higher-resolution inputs |
| Sensor anomaly detection | 100-500 ms | Driven by sensor sample rate and event rarity |
| Industrial quality inspection (line) | 50-200 ms | Driven by conveyor belt speed |
| Physical AI control loop (toy) | 50-100 ms | Driven by controller stability and actuator dynamics |
| Edge LLM assistant | 1-3 s | Tolerable because the user is in the loop |

---

## 6. Hardware + runtime

The third design axis is *where* the system runs. Chapter 2 goes deep on this; here is the system-design view.

For each candidate device, you should record:

- **Compute class:** CPU only / CPU + GPU / CPU + NPU / TPU / microcontroller.
- **Memory:** total RAM available to your process, after the OS and other services.
- **Storage:** flash / SD card / eMMC capacity for the model and logs.
- **Power envelope:** continuous wattage budget, peak wattage tolerance.
- **Thermal envelope:** is the device fan-cooled, passive-cooled, sealed in a box?
- **Connectivity:** Wi-Fi / Ethernet / cellular / none.
- **OS:** Linux distribution, Android, Windows, FreeRTOS.
- **Supported runtimes:** ONNX Runtime CPU / GPU, TensorRT, OpenVINO, TFLite, TFLite Micro, Core ML.
- **Cost** per unit and at scale.

The match between *runtime* and *hardware* matters as much as the hardware itself: a Jetson without TensorRT is not delivering its compute; a Raspberry Pi running a large PyTorch model with no quantization is wasting most of its budget.

This course's hardware × runtime × use-case mapping is in `docs/02_hardware_for_edge_ai.md`.

---

## 7. Failure modes and risk

Every Edge AI system fails. The system design should list how, and what happens when it does.

| Failure mode | Example | Mitigation |
|---|---|---|
| Sensor failure | Camera disconnected, frozen frame, dark image | Heartbeat check, sanity check on frame statistics |
| Model uncertainty | Confidence < threshold | Suppress action, log event, surface to human |
| Data drift | Lighting / camera angle / parts changed | Periodic re-evaluation, drift detector on input statistics |
| Latency spike | Thermal throttling, garbage collection, background process | Watchdog, skip frame, reduce model temporarily |
| Memory leak | Slow growth in RSS over hours | Process restart policy, monitoring |
| Network outage | Cannot reach dashboard / MQTT broker | Local buffer + replay on reconnect |
| Power loss | Brownout, battery low | Graceful shutdown, persist state |
| Wrong model deployed | Bad build, mis-tagged version | Model version + checksum logged with every prediction |
| Adversarial input | Sticker on a stop sign, manipulated frame | Out of scope for this course; flag as a known limit |

This list shows up again in Chapter 19 (security / reliability / EdgeOps). At system-design time, you only need to **name the failure modes you care about and the response for each**.

---

## 8. The system-design template

Use this template for every Edge AI application you analyze in this course. The Chapter 1 assignment fills it out for a chosen application; the final project (Ch 20) requires it as the first deliverable.

```markdown
# System design: <application name>

## 1. Problem statement
One paragraph: what is being detected / classified / decided, by whom, in what setting, with what business or operational consequence.

## 2. Input
- Source: <camera / sensor / video stream / audio / file>
- Specs: resolution / rate / format / lighting / connection
- Rate: <Hz or FPS>
- Buffering: <every frame / every N / latest only>

## 3. Output
- Type: <label / box / mask / command / alert>
- Format: <JSON schema / MQTT topic / actuator command>
- Consumer: <UI / dashboard / actuator / log>

## 4. Metric
- Quality metric + target: <e.g. recall on defective ≥ 0.95 at precision ≥ 0.90>
- Performance metric + target: <e.g. P95 end-to-end ≤ 50 ms, FPS ≥ 20>
- Cost trade-off: <false positive cost vs false negative cost>

## 5. Latency budget (P95, end-to-end)
| Stage | Budget (ms) |
|---|---|
| Capture | |
| Preprocess | |
| Model inference | |
| Postprocess | |
| Output / action | |
| Slack | |
| Total | |

## 6. Hardware + runtime
- Device: <model / SoC>
- Memory available: <MB after OS>
- Power envelope: <W>
- Runtime: <ONNX Runtime CPU / TensorRT / TFLite / ...>
- OS: <distro>

## 7. Failure modes + mitigations
| Failure | Mitigation |
|---|---|
| Sensor disconnect | |
| Low confidence | |
| Data drift | |
| Latency spike | |
| Network outage | |
| (others) | |

## 8. Out of scope
List explicitly: what the system will NOT do, what attacks it will NOT defend against, what users will NOT be supported.

## 9. Open questions
List explicitly: data you do not have, devices you have not tested on, edge cases you cannot characterize yet.
```

A complete fill-in of this template is ~1-2 pages. It should be reviewable in 5 minutes. If it takes longer, it is too long.

---

## 9. Worked example: real-time helmet detection on a Jetson Nano

To make the template concrete, here is a partial fill-in.

> **Problem:** Detect whether workers on a construction site are wearing safety helmets, in real time, from a fixed CCTV camera at the site entrance. Surface a real-time count and a "no-helmet detected" alert to the safety supervisor's dashboard.
>
> **Input:** 1920×1080 RGB H.264 stream from an IP camera at 25 FPS, RTSP over LAN. Lighting varies from bright sunlight to dusk.
>
> **Output:** A list of detections (helmet / no-helmet, box, confidence) per frame, and a "no-helmet" event when at least one no-helmet detection has confidence ≥ 0.7 sustained over 3 frames.
>
> **Metric:** Recall on "no-helmet" ≥ 0.90 at precision ≥ 0.80 (false negatives are costlier than false positives). End-to-end P95 latency ≤ 100 ms (driven by alerting responsiveness, not single-frame display). FPS_end_to_end ≥ 10.
>
> **Latency budget:** Capture+decode 30 ms / Preprocess 5 ms / Inference 40 ms / NMS+postprocess 5 ms / Event logic 5 ms / Slack 15 ms = 100 ms.
>
> **Hardware + runtime:** Jetson Nano 4GB, JetPack 5, TensorRT FP16 of a YOLOv8n exported via ONNX. Power: 10 W. Storage: 32 GB SD.
>
> **Failure modes:**
> - Camera dropout → fall back to "system offline" banner; restart RTSP after timeout.
> - Confidence < 0.5 → do not raise event; log only.
> - Thermal throttling above 75 °C → drop input to 1280×720; log warning.
> - MQTT broker unreachable → buffer last 1 hour of events locally.
>
> **Out of scope:** Face identification of workers; tracking the same worker across cameras; helmet-color compliance.
>
> **Open questions:** Performance at dusk vs daylight has not been measured; only 200 labeled "no-helmet" examples available.

A learner who can write this template confidently for their own problem is ready for Chapter 4 (benchmarking) and Chapter 9-10 (camera AI implementation).

---

## 10. What you should be able to do after this chapter

- Describe any Edge AI application as a system with input, preprocessing, inference, postprocessing, and output stages.
- Pick a quality metric *and* a performance metric *and* an accuracy-vs-latency trade-off, not just one.
- Set a latency budget *before* picking a model.
- Allocate latency across pipeline stages and identify which stage is over budget.
- List at least five failure modes for your chosen application and a mitigation for each.
- Fill in the system-design template in `assignments/assignment_01_edge_ai_analysis.md`.

---

## 11. Files produced by this chapter

- `docs/01_edge_ai_system_design.md` — this file.
- `assignments/assignment_01_edge_ai_analysis.md` — assignment that requires filling the template above for a chosen application.
