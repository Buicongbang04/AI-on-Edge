# Edge AI and Introductory Physical AI

**From On-device Inference to Perception-Action Loops**

A complete course repository teaching how to take AI models from a notebook/server/GPU training environment to a real edge device — running stably under real-world constraints: latency, memory, power, temperature, cost, privacy, and reliability.

The course is structured in **5 parts**, **20 chapters**, and a **14-week** plan, plus an optional 2-week extension for advanced topics (Edge LLM/VLM) and the final capstone project.

---

## Who this course is for

| Audience | What they get from this course |
|---|---|
| AI Engineer | Practice taking a model to production on real hardware, optimizing latency and memory |
| AI / Computer Vision / IoT student | Build a camera AI, sensor AI, robot simulator, or TinyML project |
| Computer Vision Engineer | Deploy YOLO / classifier / segmentation models on edge devices |
| Embedded / IoT Engineer | Combine AI inference with camera, sensor, MQTT, dashboard, or actuator |
| Anyone interested in Physical AI | Learn the perception-action loop fundamentals before deeper robotics work |

**Minimum prerequisites:** Python, NumPy, Git, basic Linux command line, PyTorch basics (model, checkpoint, dataloader, train/eval mode). Familiarity with OpenCV is a plus. No edge hardware required to start — the repo runs on a laptop CPU/GPU at level 1.

---

## What you will be able to do after the course

- Distinguish training, inference, and deployment, and reason about latency / memory / power / bandwidth / cost / reliability trade-offs.
- Load a model, run inference in PyTorch and ONNX Runtime, export to ONNX / TorchScript / TFLite, and validate that outputs match.
- Apply FP16 / INT8 quantization (PTQ, intro QAT), pruning, and distillation, and measure the accuracy-vs-latency trade-off.
- Benchmark **end-to-end** (capture → preprocess → inference → postprocess → output) — reporting P50/P95/P99 latency, FPS, peak memory, and utilization, not just model-only latency.
- Build a real-time camera classifier and a YOLO real-time detector on a laptop / Jetson, with FPS counter, frame skipping, and threading.
- Build a sensor anomaly detection pipeline with windowing and a TinyML-style demo (concept-level for users without a microcontroller board).
- Design a Physical AI loop: **perception → state → decision → controller → action → feedback**, with a safety gate before any actuator command.
- Write deployment notes, an error analysis log, a rollback plan, and risk/safety notes for an edge AI system.

---

## Repository layout

```
edge-ai-physical-ai-systems/
├── README.md                  - this file
├── COURSE_OVERVIEW.md         - 5-part structure, target audience, what you build
├── SYLLABUS.md                - chapter-by-chapter syllabus (Ch 0-20)
├── ROADMAP.md                  - 14-week plan + deliverable per week
├── requirements.txt           - pip dependencies
├── environment.yml            - conda environment (Python 3.11)
├── docs/                      - one markdown file per chapter (00-20)
├── notebooks/                 - executable notebooks per chapter
├── labs/                      - guided labs with checkpoints
├── assignments/               - graded assignments
├── projects/                  - 6 mini-projects + final_project_template
├── src/                       - reusable code (inference/export/optimization/benchmark/...)
├── configs/                   - YAML configs for projects
├── hardware_notes/            - per-hardware setup notes (laptop/RPi/Jetson/Intel)
├── datasets/                  - small datasets or download scripts
├── experiments/               - logs, checkpoints, exported models, benchmark results
├── deployment_notes/          - runtime.md, safety.md, hardware.md per project
├── figures/                   - diagrams for docs
└── reports/                   - report templates and example reports
```

See [`COURSE_OVERVIEW.md`](COURSE_OVERVIEW.md) for the full curriculum structure and [`SYLLABUS.md`](SYLLABUS.md) for the chapter list.

---

## Quick start

```bash
# 1. Clone the repo
git clone <your-fork-url> edge-ai-physical-ai-systems
cd edge-ai-physical-ai-systems

# 2. Create the conda environment (Python 3.11)
conda env create -f environment.yml
conda activate edge-ai

# (or, if you already have an env, just install requirements)
pip install -r requirements.txt

# 3. Run the first inference demo (Chapter 3)
python src/inference/infer_pytorch.py --image datasets/sample.jpg

# 4. Open the first notebook
jupyter lab notebooks/chapter_01_latency_benchmarking.ipynb
```

The repo is designed to run at **Level 1** on a laptop CPU or GPU. Sections that require Jetson, Raspberry Pi, Intel NPU, Google Coral, or a microcontroller are marked clearly and include a fallback that works on a laptop.

---

## How to use this repo

1. **Read the syllabus** to pick where to start. If you are new to deployment, follow chapters sequentially. If you already deploy models, you can skip to the optimization (Ch 8), camera (Ch 9-10), or Physical AI (Ch 14-17) sections.
2. **Each chapter has**: a doc page (`docs/NN_*.md`), often a notebook or lab, sometimes an assignment, and reusable code in `src/`.
3. **Each project has**: its own `README.md`, a `config.yaml`, and scripts for `train` / `export` / `infer` / `benchmark`, plus a `results/` directory.
4. **Benchmarks must be end-to-end.** Never report "the model runs at X ms" if your real pipeline also includes capture, resize, normalize, postprocess, draw overlay, send alert, or publish a command. The end-to-end latency template is in `src/benchmark/`.
5. **For the final project**, copy `projects/final_project_template/` and fill in the 12 required sections (problem, device, dataset, baseline, export, runtime, benchmark, metric, demo, error analysis, deployment notes, safety notes). The rubric is 100 points across 10 criteria — see `reports/final_project_report_template.md`.

---

## Course philosophy

- **The model is only one piece of the system.** A model with 99% accuracy but 2 GB of RAM, 800 ms of latency, or thermal throttling at room temperature is not deployable.
- **Measure before you optimize.** Always benchmark model size, latency P50/P95, FPS, RAM, utilization, power, and temperature *where it is measurable* — before applying quantization or pruning.
- **Start from a simple pipeline.** Build `camera → frame → preprocess → infer → postprocess → output` before adding a robot, a controller, or any closed loop.
- **Accuracy / latency / power is a triangle.** A real edge project must state which two corners it prioritizes — not just report accuracy.
- **Physical AI needs an action loop.** A model that classifies is not yet a Physical AI system; Physical AI requires sensor input, state estimation, a decision, an action, a feedback signal, and a safety boundary.
- **Safety and fallback are mandatory.** When an AI system touches the physical world, it needs a rule-based fallback, a human override path, and a safety layer in front of the actuator.

---

## License & contributions

Educational use. Forks, issues, and pull requests are welcome — please open an issue first to discuss substantial changes.
