# Chapter 20 — Final project end-to-end

> **Goal:** Demonstrate end-to-end deployment competence on a chosen problem. Produce a reproducible repo project with benchmark, demo, deployment notes, and safety notes. This is the capstone of the course.

This chapter assembles everything from Parts 1-4 plus the EdgeOps discipline of Chapter 19 into one complete project. The grading rubric (100 points across 10 criteria) is at the end of this document and lives in `reports/final_project_report_template.md`.

---

## 1. Choosing your project

Pick one of the suggested templates *or* propose your own. The project must:

- Run on real hardware OR in a simulator (camera AI, sensor AI, TinyML, Physical AI simulation).
- Have a meaningful end-to-end pipeline — not just `model(x)`.
- Be benchmarked end-to-end (not just model-only).
- Have a deployment plan and a safety plan, even if the deployment is "laptop only".

### Suggested project templates

1. **Real-time camera classifier** on a laptop or Jetson (extends `project_01_camera_classifier`).
2. **YOLO real-time object detection** — people counting, PPE, vehicle detection (extends `project_02_yolo_realtime_camera`).
3. **Product defect detection** — MVTec AD or your own dataset (extends `project_04_quality_inspection_ai`).
4. **Sensor anomaly detection** — vibration / temperature / current pipeline (extends `project_03_sensor_anomaly_detection`).
5. **TinyML keyword spotting or gesture recognition** — concept or real MCU (extends `project_05_tinyml_keyword_spotting`).
6. **Edge dashboard for camera AI** — a complete operator UI on top of an inference loop.
7. **Physical AI simulator** — extend `project_06_physical_ai_simulation` with a richer environment, perception, or a learned policy.
8. **Edge LLM assistant for camera monitoring** — bonus / advanced — combine VLM + small LLM for a camera-driven assistant.

If your project is custom, run the choice past the instructor / mentor before starting — to avoid scope mismatch.

---

## 2. The required deliverables

Every final project must produce:

```
projects/<your_project>/
├── README.md               — problem, hardware, dataset, model, results
├── config.yaml             — runtime configuration
├── train.py (or link)      — how the model was trained
├── export.py (or link)     — model → ONNX / TFLite
├── infer.py                — runtime inference script (camera / video / stream)
├── benchmark.py            — calls src.benchmark with project-specific inputs
├── results/                — benchmark JSON, plots, demo video, screenshots
└── deployment_notes/
    ├── runtime.md          — see template in repo root deployment_notes/
    └── safety.md           — see template in repo root deployment_notes/
```

Plus one **report** at `reports/<your_project>_report.md` using the rubric template at `reports/final_project_report_template.md`.

---

## 3. The 12 required report sections

These are required by the rubric. Use the template at `reports/final_project_report_template.md` — do not omit sections.

1. **Problem statement** — who, what, why, with what consequence.
2. **Device / simulator environment** — exact hardware, runtime, software stack.
3. **Dataset** — source, size, splits, labeling rules.
4. **Model baseline** — architecture, training summary, FP32 accuracy.
5. **Export / optimization** — ONNX path, quantization, validation diff vs PyTorch.
6. **Runtime** — chosen runtime + execution provider + threading.
7. **Benchmark** — model-only latency, end-to-end latency, FPS, memory, JSON in `experiments/benchmark_results/`.
8. **Metric** — quality metric AND performance metric AND cost trade-off.
9. **Demo script** — what command produces the demo, how to verify the output.
10. **Error analysis** — failure cases, confusion matrix, distribution shifts.
11. **Deployment notes** — link to `deployment_notes/runtime.md`.
12. **Risk / safety notes** — link to `deployment_notes/safety.md`.

---

## 4. Pipeline expectations

The rubric requires the full pipeline:

| Stage | Required? | Where |
|---|---|---|
| Problem + constraint statement | Yes | report §1 |
| Dataset + preprocessing | Yes | report §3 + `data/` |
| Model baseline (any reasonable) | Yes | report §4 |
| Export to ONNX (or chosen format) | Yes | `export.py` |
| Optimization (quantization at minimum) | Recommended | report §5 |
| Runtime + execution provider | Yes | report §6 + `config.yaml` |
| End-to-end benchmark | Yes | `benchmark.py` + `experiments/benchmark_results/` |
| Evaluation metric appropriate to task | Yes | report §8 |
| Working demo | Yes | `infer.py` + `results/demo.*` |
| Error analysis | Yes | report §10 |
| Deployment notes (hardware, runtime) | Yes | `deployment_notes/runtime.md` |
| Risk and safety notes | Yes | `deployment_notes/safety.md` |

---

## 5. Grading rubric (100 points)

(matches Instruction.pdf §16)

| Criterion | Points |
|---|---|
| Problem statement and edge constraints are clearly stated | 10 |
| Dataset and preprocessing are correct | 10 |
| Model baseline is reasonable | 10 |
| Export / optimization / inference runtime are documented | 15 |
| Benchmark latency / FPS / memory / end-to-end is honest | 15 |
| Evaluation metric is appropriate to the task | 10 |
| Demo runs | 10 |
| Error analysis is present | 10 |
| Deployment notes and safety notes are written | 5 |
| Repo is clean and reproducible | 5 |
| **Total** | **100** |

---

## 6. Anti-patterns that lose points

- Reporting only `mean` latency, never `P95`.
- Reporting only `accuracy` on an imbalanced task.
- "Optimized the model" with no before/after numbers.
- No end-to-end FPS — only model-only latency.
- Missing `deployment_notes/safety.md`.
- A demo that does not actually run because the repo state assumed paths or weights you did not commit.
- Vague "future work" sections instead of concrete error analysis on the current model.

---

## 7. Bonus credit (optional)

- **Sustained-load benchmark** (≥5 min) showing whether latency drifts.
- **Cross-hardware comparison** (e.g. laptop CPU vs Jetson) with the same model.
- **Real OTA deployment** (Mender, balena, or shell script) of a model update with rollback.
- **Operator dashboard** (Streamlit, Grafana, or a small Flask app) reading from the device log.

These are not graded but stand out in a portfolio.

---

## 8. What you should be able to do after this chapter

- Pick a real edge AI problem and execute it end-to-end.
- Justify every design choice with a number.
- Produce a clean, reviewable repo.
- Write deployment and safety notes that a real operator could act on.

---

## 9. Files produced by this chapter

- `docs/20_final_project.md` — this file.
- `projects/final_project_template/` — empty template that students fork.
- `reports/final_project_report_template.md` — the report template the rubric grades against.
