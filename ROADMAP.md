# 14-week roadmap

This roadmap maps the 20 chapters in [`SYLLABUS.md`](SYLLABUS.md) onto a 14-week course schedule, with a recommended deliverable for each week.

A 2-week optional extension is available for Edge LLM / multimodal AI and the deeper Physical AI lab work. The total runs 14 weeks for the core course and up to 16 weeks with extensions.

---

## Weekly plan

| Week | Topic (chapters) | Deliverable | Difficulty |
|---|---|---|---|
| 1 | Edge AI / Physical AI overview (Ch 0) | Short concept note: distinguish Edge AI vs Physical AI vs Cloud AI; one-page application map | Light |
| 2 | Edge AI system design + latency budget (Ch 1) | Assignment: system-design analysis of a chosen Edge AI application — input, output, metric, latency target, failure modes | Light |
| 3 | Hardware for Edge AI (Ch 2) | Hardware comparison report + runtime selection note (which device fits which problem and why) | Light |
| 4 | Inference basics, benchmarking foundations (Ch 3, Ch 4 start) | PyTorch inference script + basic benchmark script reporting mean and P95 latency | Medium |
| 5 | PyTorch → ONNX export (Ch 5) | Export a small model to ONNX; validate output equivalence vs PyTorch | Medium |
| 6 | ONNX Runtime (Ch 6) | ONNX inference demo + CPU/GPU benchmark comparison; reproduce vs PyTorch baseline | Medium |
| 7 | TensorRT / OpenVINO / TFLite overview (Ch 7) | Runtime selection note; optional hands-on lab if the matching hardware is available | Medium |
| 8 | Quantization, pruning, distillation (Ch 8) | Optimization report: size, latency, accuracy before/after — and which trade-off was accepted | Medium |
| 9 | Real-time camera AI (Ch 9) | Camera classifier demo with FPS counter + screenshot of overlay + benchmark on the laptop or Jetson | Medium |
| 10 | Object detection on the edge (Ch 10) | YOLO real-time camera demo + ONNX export of YOLO + FPS benchmark | Medium |
| 11 | Sensor AI / TinyML (Ch 11, Ch 12) | Sensor anomaly detection demo OR TinyML keyword-spotting concept demo | Medium |
| 12 | Physical AI loop (Ch 14, Ch 15, Ch 16) | Perception → state → decision → action simulation loop with safety gate, logged to file | Hard |
| 13 | Safety, reliability, EdgeOps + Edge LLM (Ch 17, Ch 18, Ch 19) | Deployment design + risk checklist + log schema for your chosen project | Hard |
| 14 | Final project presentation (Ch 20) | Final project: report + working demo + benchmark + deployment notes + safety notes | Hardest |

---

## Optional weeks 15-16

- **Week 15:** Edge LLM hands-on (Ch 18 deep dive) — run a quantized small language model locally; measure prompt latency and memory; sketch a camera → vision model → local LLM decision pipeline.
- **Week 16:** Physical AI deeper lab — extend the Ch 14-16 simulation with a learned policy (toy RL or imitation) or add a more realistic controller. Optional ROS2 integration (Ch 17) if installed.

---

## Pacing notes

- **Weeks 1-3 are light by design.** They establish the systems mindset. Do not skip them; they shape every later design choice.
- **Weeks 4-8 are the most code-heavy on the deployment side.** This is where PyTorch → ONNX → runtime → quantization gets wired up. Save your benchmark numbers — they reappear in the final project.
- **Weeks 9-11 are application weeks.** Pick the application track most relevant to you; you do not have to do all three (camera, detection, sensor) at the same depth.
- **Weeks 12-13 are Physical AI weeks.** These rely on simulation rather than physical hardware — the toy simulation is intentional and runs in Python only.
- **Week 14 is presentation week.** By then your final project should already be working; week 14 is for polishing the report, demo script, and notes.

---

## Recommended weekly time budget

For a learner who already has the prerequisites:

- Reading + viewing: 2 hours
- Hands-on lab / notebook: 3-4 hours
- Project / assignment work: 3-4 hours

Total: ~8-10 hours per week. Final project weeks (13-14) may need 12-15 hours total.

---

## How to use the roadmap

1. Set a weekly deliverable target — the table above gives one suggestion per week.
2. After each week, save your deliverable into `experiments/` (benchmark numbers) or your forked `projects/` (code).
3. Maintain a running benchmark sheet so you can compare optimizations against each other and against the baseline.
4. Treat the final project as a checkpoint, not an afterthought: pick a topic by week 8 and start collecting data and notes from week 9.
