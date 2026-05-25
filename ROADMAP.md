# 14-week roadmap

This roadmap maps the 20 chapters listed in [`SYLLABUS.md`](SYLLABUS.md) onto a 14-week schedule, with a concrete weekly deliverable and the exact files in this repo that a learner should consume that week.

A 2-week optional extension covers Edge LLM hands-on (Ch 18 deeper) and an optional Physical AI / RL extension. Total runs 14 weeks for the core, up to 16 with extensions.

For the introduction and how-to-learn guidance, start with [`README.md`](README.md). For the part-level rationale, see [`COURSE_OVERVIEW.md`](COURSE_OVERVIEW.md).

---

## How a week works

Each week has the same four-step pattern:

1. **Read** the chapter doc(s) in `docs/`.
2. **Run** the chapter notebook(s) in `notebooks/` (and any per-chapter project script).
3. **Do** the assignment (if the chapter has one) and / or the lab.
4. **Save** the produced artifacts in `experiments/` (logs, benchmark JSONs) or in your forked `projects/` (code, demos).

Treat the deliverable as a checkpoint. Saving JSONs and demos as you go is how you assemble the final-project report (week 14) without scrambling.

---

## Weekly plan

| Week | Topic (chapters) | Read | Run | Deliverable | Difficulty |
|---|---|---|---|---|---|
| 1 | Edge AI / Physical AI overview (Ch 0) | `docs/00_intro_edge_physical_ai.md`, `figures/cloud_vs_edge.png`, `figures/physical_ai_loop.png` | n/a | One-page concept note: distinguish Edge AI, Physical AI, Cloud AI, plus an application map of 3-5 ideas you care about | Light |
| 2 | Edge AI system design + latency budget (Ch 1) | `docs/01_edge_ai_system_design.md` | n/a | `assignments/assignment_01_edge_ai_analysis.md` filled in for a chosen application | Light |
| 3 | Hardware for Edge AI (Ch 2) | `docs/02_hardware_for_edge_ai.md`, all four `hardware_notes/*.md` | n/a | Hardware comparison table + runtime selection note (which device fits which problem and why) | Light |
| 4 | Inference basics + benchmarking foundations (Ch 3, Ch 4 start) | `docs/03_model_inference_basics.md`, `docs/04_benchmarking_and_profiling.md` | `notebooks/chapter_01_latency_benchmarking.ipynb`, `python -m src.benchmark --model mobilenet_v3_small --device cpu` | PyTorch inference script + first benchmark JSON in `experiments/benchmark_results/` | Medium |
| 5 | PyTorch -> ONNX export (Ch 5) | `docs/05_pytorch_to_onnx.md` | `notebooks/chapter_02_pytorch_to_onnx.ipynb`, `labs/lab_02_export_pytorch_to_onnx.ipynb`, `python src/export/export_onnx.py --model mobilenet_v3_small` | Exported `.onnx` in `experiments/exported_models/`; numerical diff against PyTorch reported | Medium |
| 6 | ONNX Runtime (Ch 6) | `docs/06_onnx_runtime.md` | `notebooks/chapter_03_onnxruntime_inference.ipynb`, `python src/inference/infer_onnxruntime.py --model <onnx> --image datasets/sample.jpg` | ORT vs PyTorch benchmark comparison; one row per provider you have | Medium |
| 7 | TensorRT / OpenVINO / TFLite overview (Ch 7) | `docs/07_tensorrt_openvino_tflite.md`, all four `hardware_notes/*.md` | `labs/lab_04_tensorrt_or_openvino.md` (only if you have the matching hardware) | Runtime selection note + optional hands-on table | Medium |
| 8 | Quantization, pruning, distillation (Ch 8) | `docs/08_model_optimization.md` | `notebooks/chapter_04_quantization_ptq.ipynb` | `assignments/assignment_04_quantization.md` filled in: size / latency / accuracy table for FP32 vs INT8 dynamic vs INT8 static | Medium |
| 9 | Real-time camera AI (Ch 9) | `docs/09_realtime_camera_ai.md` | `notebooks/chapter_06_camera_inference_opencv.ipynb`, `python src/inference/camera_loop.py --model <onnx>`, `labs/lab_05_realtime_camera.md` | Camera classifier demo with FPS counter + screenshot or recording; benchmark on laptop and (if available) Jetson / Pi | Medium |
| 10 | Object detection on the edge (Ch 10) | `docs/10_object_detection_edge.md`, `projects/project_02_yolo_realtime_camera/README.md` | `notebooks/chapter_07_yolo_edge_detection.ipynb`; the three project scripts (`run_image.py`, `run_video.py`, `export_onnx.py`) | YOLO real-time demo + exported ONNX + FPS benchmark | Medium |
| 11 | Sensor AI + TinyML (Ch 11, Ch 12) | `docs/11_sensor_ai_timeseries.md`, `docs/12_tinyml_microcontrollers.md` | `notebooks/chapter_08_sensor_anomaly_detection.ipynb` + the three `project_03_sensor_anomaly_detection/` scripts; `notebooks/chapter_09_tinyml_intro.ipynb` + `project_05_tinyml_keyword_spotting/` scripts | Either: sensor anomaly demo with detected events JSON, OR TinyML KWS concept demo with footprint table | Medium |
| 12 | Physical AI loop (Ch 14, Ch 15, Ch 16) | `docs/14_physical_ai_loop.md`, `docs/15_control_safety_feedback.md`, `docs/16_simulation_sim_to_real.md`, `projects/project_06_physical_ai_simulation/README.md` | `notebooks/chapter_10_physical_ai_control_loop.ipynb`; `python projects/project_06_physical_ai_simulation/run_simulation.py`; `labs/lab_08_robot_simulation_loop.md` | Perception -> state -> decision -> action simulation loop with safety gate; `event_log.json` + `trajectory.png` saved | Hard |
| 13 | Safety, reliability, EdgeOps, Edge LLM intro (Ch 17, Ch 18, Ch 19) | `docs/17_ros2_robot_sim_intro.md`, `docs/18_edge_llm_multimodal_ai.md`, `docs/19_security_reliability_edgeops.md` | `notebooks/chapter_11_edge_llm_intro.ipynb`; `labs/lab_09_mock_ros_loop.md`; smoke-test `src/edgeops/` helpers | Deployment design + risk checklist + device log schema for your chosen project (use `deployment_notes/runtime.md` and `safety.md` templates) | Hard |
| 14 | Final project presentation (Ch 20) | `docs/20_final_project.md`, `projects/final_project_template/README.md`, `reports/final_project_report_template.md` | Whatever your project needs | Final project: report + working demo + benchmark JSONs + deployment notes + safety notes; graded by the 100-point rubric | Hardest |

---

## Optional weeks 15-16

- **Week 15 — Edge LLM hands-on (deeper Ch 18).** Pick a small quantized model (e.g. via `ollama pull qwen2.5:0.5b-instruct`) and reproduce the memory math from `chapter_11_edge_llm_intro.ipynb` against your laptop. Measure TTFT and tokens/second.
- **Week 16 — Physical AI extension.** Extend `project_06_physical_ai_simulation/` with a richer environment (more obstacles, dynamic targets), a noisier perception (raise `noise_std`), or a learned policy (toy RL or imitation). Optional ROS2 integration if installed.

---

## Pacing notes

- **Weeks 1-3 are light by design.** They establish the systems mindset that the rest of the course assumes. Do not skip them — they shape every later design choice.
- **Weeks 4-8 are the most code-heavy on the deployment side.** This is where PyTorch -> ONNX -> runtime -> quantization gets wired up. Save your benchmark JSONs — they reappear in the final project.
- **Weeks 9-11 are application weeks.** Pick the application track most relevant to you (camera / detection vs sensor / TinyML); you do not have to do all three at the same depth.
- **Weeks 12-13 are Physical AI weeks.** These rely on simulation rather than physical hardware — the toy simulation is intentional and runs in Python only.
- **Week 14 is presentation week.** By then your final project should already be working; week 14 is for polishing the report, demo script, and notes.

---

## Recommended weekly time budget

For a learner with the prerequisites: roughly 2 hours reading + 3-4 hours hands-on lab + 3-4 hours project work per week. Total ~8-10 hours/week, with weeks 13-14 needing 12-15 hours total.

---

## Suggested deliverable layout (so your final project is easy in week 14)

By the end of each week, drop the produced artifact into one of:

- `experiments/benchmark_results/<name>-<timestamp>.json` for benchmark runs.
- `experiments/reports/assignment_NN_<your_name>.md` for graded assignments.
- `projects/<your_project>/results/` for your project demos, plots, and JSONs.
- `deployment_notes/runtime.md` and `safety.md` within your project (copy from the repo-root templates).

By week 14 you should already have everything you need; the final report (`reports/<your_project>_report.md`) becomes a write-up of work already done, not new work.
