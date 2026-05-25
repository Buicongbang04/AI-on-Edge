# Edge AI and Introductory Physical AI

*From On-device Inference to Perception-Action Loops*

A complete course repository covering the path from a trained model to a deployed edge AI system, and on to the perception-action loops of introductory Physical AI. The course is organized in **5 parts**, **20 chapters**, and a **14-week plan**, with all code runnable on a laptop CPU (Level 1) and optional hands-on for Jetson, Raspberry Pi, Intel NPU, and microcontroller class hardware.

---

## What this course is, in one paragraph

By the end of the course you will be able to take a PyTorch model, export it to ONNX, optimize it with quantization, run it via ONNX Runtime (and optionally TensorRT, OpenVINO, or TFLite), measure its end-to-end latency / FPS / memory honestly, integrate it with a webcam or sensor pipeline, design a real-time camera AI or sensor-anomaly system, build a perception-action loop with a safety gate for introductory Physical AI, and write the deployment notes / logging / fallback / rollback discipline that turn a demo into a deployable system.

---

## Who this course is for

| Audience | What you get |
|---|---|
| AI / ML engineer | Practice taking a model to production on real hardware, optimizing latency and memory |
| Student in AI / CV / IoT | Build six mini-projects spanning camera, sensor, TinyML, quality inspection, and a Physical AI simulation |
| Computer vision engineer | Deploy classifier / YOLO / segmentation on edge devices with proper benchmarking |
| Embedded / IoT engineer | Combine AI inference with camera, sensor, MQTT, dashboard, or actuator |
| Anyone curious about Physical AI | Learn perception-state-decision-action loops before deeper robotics work |

**Minimum prerequisites:** Python, NumPy, Git, basic Linux command line, PyTorch basics (model, checkpoint, dataloader, train/eval mode). Familiarity with OpenCV is a plus. **No edge hardware required to start** — every chapter runs at Level 1 on a laptop CPU.

---

## How to use this repository

The repo is both a textbook (the `docs/` chapters) and a working toolkit (the `src/` modules, `notebooks/`, and `projects/`). You can read it linearly or pick a track.

### Recommended path: linear

If you are new to deployment, follow the chapters in order. Each chapter has:

- a **doc** in `docs/NN_<topic>.md` explaining the concepts,
- usually a **notebook** in `notebooks/chapter_NN_<topic>.ipynb` walking through the doc with executable code,
- sometimes a **lab** in `labs/` (more guided, with exercises),
- sometimes an **assignment** in `assignments/` (graded, see rubric),
- and reusable **code** in `src/`.

A typical week looks like: read the chapter doc, run the notebook, do the assignment (if any), then attempt the lab if there is one. Reference scripts in `src/` are what you reuse in your own projects.

### Alternative paths (if you already know parts of the stack)

| Goal | Suggested chapter order |
|---|---|
| Deploy YOLO on Jetson | Ch 1, 2, 4, 5, 6, 7, 10 |
| Sensor anomaly detection / TinyML | Ch 1, 4, 8, 11, 12 |
| Understand Physical AI | Ch 0, 1, 9, 14, 15, 16, 17 |
| Jump straight to the final project | Ch 0, 1, 4, 19, 20 |

### Recommended weekly time budget

For a learner with the prerequisites: roughly 2 hours of reading + 3-4 hours of hands-on lab + 3-4 hours of project work per week. Total ~8-10 hours/week, with weeks 13-14 likely needing 12-15 hours for the final project.

---

## Quick start

```bash
# 1. Clone and enter the repo
git clone <your-fork-url>
cd Edge_Physical_AI

# 2. Create the conda environment (Python 3.11)
conda env create -f environment.yml
conda activate edge-ai
#  -- or, to reuse an existing env, install requirements directly --
# pip install -r requirements.txt

# 3. Run your first PyTorch inference (Chapter 3)
python src/inference/infer_pytorch.py --image datasets/sample.jpg

# 4. Export the model to ONNX (Chapter 5)
python src/export/export_onnx.py --model mobilenet_v3_small

# 5. Run the same model via ONNX Runtime and benchmark it (Chapter 4 + 6)
python -m src.benchmark --model mobilenet_v3_small --device cpu --pipeline

# 6. Open the first notebook
jupyter lab notebooks/chapter_01_latency_benchmarking.ipynb
```

The whole repo is designed to run at **Level 1** on a laptop CPU or GPU. Chapters that require specific edge hardware (Jetson for TensorRT, Intel NPU for OpenVINO, microcontroller for TFLite Micro) are clearly marked, and the per-hardware setup notes live in `hardware_notes/`.

---

## Repository layout

```
.
├── README.md                  - this file
├── COURSE_OVERVIEW.md         - 5-part structure, audience, philosophy
├── SYLLABUS.md                - chapter-by-chapter syllabus with file references
├── ROADMAP.md                 - 14-week plan and weekly deliverables
├── requirements.txt           - pip dependencies
├── environment.yml            - conda environment (Python 3.11)
│
├── docs/                      - one markdown file per chapter (00..20)
├── notebooks/                 - 10 executable notebooks (chapter_01..chapter_11; chapter_05 has no notebook)
├── labs/                      - 5 guided labs with exercises
├── assignments/               - 3 graded assignments (Ch 1, 4, 8)
├── projects/                  - 6 mini-projects + final_project_template
│
├── src/                       - reusable code
│   ├── inference/             - PyTorch, ONNX Runtime, TensorRT, OpenVINO, camera loop
│   ├── export/                - PyTorch -> ONNX export with validation
│   ├── optimization/          - INT8 quantization (dynamic + static) helpers
│   ├── benchmark/             - latency / memory / FPS benchmark suite + CLI
│   ├── physical_ai/           - perception, decision, controller, safety modules
│   └── edgeops/               - device logging, model versioning, fallback
│
├── configs/                   - per-project YAML configs (empty placeholder)
├── hardware_notes/            - per-hardware setup notes (laptop, RPi, Jetson, Intel)
├── datasets/                  - sample inputs + scripts that generate synthetic data
├── experiments/               - logs, checkpoints, exported models, benchmark JSON
├── deployment_notes/          - runtime.md and safety.md templates
├── figures/                   - diagrams for the chapter docs
└── reports/                   - report templates (incl. final project rubric)
```

For the chapter-by-chapter list with exact file references, see [`SYLLABUS.md`](SYLLABUS.md).

For the weekly schedule with deliverables, see [`ROADMAP.md`](ROADMAP.md).

---

## What you will be able to do after the course

- Distinguish training, inference, and deployment, and reason about latency / memory / power / bandwidth / cost / reliability trade-offs.
- Load a model, run inference in PyTorch and ONNX Runtime, export to ONNX / TorchScript / TFLite, and validate that outputs match.
- Apply FP16 / INT8 quantization (PTQ, intro QAT), pruning, and distillation, and measure the accuracy-vs-latency trade-off honestly.
- Benchmark **end-to-end** (capture, preprocess, inference, postprocess, output) and report P50 / P95 / P99 latency, FPS, peak memory, and utilization.
- Build a real-time camera classifier and a YOLO real-time detector on a laptop or Jetson, with FPS counter, frame skipping, and recording.
- Build a sensor anomaly detection pipeline with windowing and an autoencoder-based detector.
- Estimate the memory footprint of a TinyML model and decide which microcontroller it fits on.
- Design a Physical AI loop (perception, state, decision, controller, action, feedback) with a safety gate before any actuator command.
- Write deployment notes, an error analysis, a rollback plan, and risk / safety notes for an edge AI system.

---

## Course philosophy

- **The model is only one piece of the system.** A model with 99% accuracy but 2 GB of RAM, 800 ms of latency, or thermal throttling at room temperature is not deployable.
- **Measure before you optimize.** Always benchmark size, latency P50 / P95, FPS, RAM, utilization, power, and temperature (where measurable) before applying quantization or pruning.
- **Start from a simple pipeline.** Build `capture -> preprocess -> inference -> postprocess -> output` before adding a robot, a controller, or any closed loop.
- **Accuracy, latency, and power is a triangle.** A real edge project must state which two corners it prioritizes, not just report accuracy.
- **Physical AI needs an action loop.** Perception-only AI is not yet Physical AI; Physical AI requires sensor input, state estimation, decision, action, feedback, and a safety boundary.
- **Safety and fallback are mandatory.** When an AI system touches the physical world, it must have a rule-based fallback, a human override path, and a safety layer in front of any actuator.

---

## Hardware support tiers

Not every learner owns every kind of device. The repo supports a fallback ladder:

| Tier | Hardware | What works |
|---|---|---|
| Level 1 (minimum) | Laptop CPU or GPU | The whole repo: PyTorch and ONNX Runtime inference, camera demo via OpenCV, quantization, sensor pipelines, Physical AI simulation, Edge LLM concept |
| Level 2 | Raspberry Pi 5 / Intel NUC | TFLite / ONNX Runtime / OpenVINO inference; realistic edge benchmarks |
| Level 3 | NVIDIA Jetson Orin Nano (or NX) | TensorRT, real-time YOLO at 30+ FPS, FP16/INT8 acceleration, multi-stream camera AI |
| Level 4 | Microcontroller (Arduino BLE, ESP32, etc.) | TinyML via TensorFlow Lite Micro |
| Optional | Google Coral TPU | TFLite Edge TPU acceleration |

The repo *always* runs at Level 1. Higher-tier sections are clearly marked, and their setup notes live in `hardware_notes/`.

---

## Help, contributions, license

- For setup or content questions, open an issue in your fork.
- For improvements (typos, clearer explanations, new use cases): pull requests welcome.
- Educational use. The repo bundles small synthetic datasets only; for real datasets (ImageNet, MVTec AD, Google Speech Commands, etc.), follow the original dataset's license.
