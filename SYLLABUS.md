# Syllabus

Chapter-by-chapter syllabus for **Edge AI and Introductory Physical AI**. Each chapter lists its goal, main topics, learning outcomes, and the exact files in this repo that it ships.

For the 5-part rationale, see [`COURSE_OVERVIEW.md`](COURSE_OVERVIEW.md). For the week-by-week plan with deliverables, see [`ROADMAP.md`](ROADMAP.md). For the high-level intro and quick-start, see [`README.md`](README.md).

---

## Part 1 — Edge AI Foundation

### Chapter 0 — Edge AI and Physical AI overview

- **Goal:** Distinguish Edge AI vs Cloud AI, define Physical AI, and motivate why this direction matters.
- **Topics:** Cloud AI vs Edge AI; on-device inference; real-time AI; latency, throughput, FPS; privacy and offline inference; Physical AI (perception, state, decision, action); use cases.
- **Outcomes:** Explain Edge AI vs Physical AI in your own words; distinguish from Embedded AI / TinyML / Robotics AI; cite real-world examples per application group.
- **Files:**
    - `docs/00_intro_edge_physical_ai.md`
    - `figures/cloud_vs_edge.png`, `figures/physical_ai_loop.png`
    - `figures/_generate_chapter_00_figures.py` (script that produces the figures)

### Chapter 1 — Edge AI system design

- **Goal:** Frame an Edge AI problem as a system, not a model choice.
- **Topics:** Input source (camera / mic / sensor / video); preprocessing / inference / postprocessing; output (label / box / mask / command / alert); latency budget; accuracy requirement; hardware / power / deployment environment; failure modes.
- **Outcomes:** Write a one-page system-design note for a chosen Edge AI application; specify input, output, metric, latency target, hardware, and risks.
- **Files:**
    - `docs/01_edge_ai_system_design.md`
    - `assignments/assignment_01_edge_ai_analysis.md` (template + 100-point rubric)

### Chapter 2 — Hardware for Edge AI

- **Goal:** Map application classes to hardware classes.
- **Topics:** CPU / GPU / NPU / TPU accelerators; microcontroller vs Raspberry Pi vs NVIDIA Jetson vs Intel CPU / NPU vs Google Coral TPU; camera modules and sensors; cost / power / compute / ecosystem trade-offs.
- **Outcomes:** Distinguish microcontroller, SBC, and edge GPU classes; pick the right device class for a given task; know which runtime fits which device.
- **Files:**
    - `docs/02_hardware_for_edge_ai.md`
    - `hardware_notes/laptop_cpu_gpu.md`
    - `hardware_notes/raspberry_pi.md`
    - `hardware_notes/nvidia_jetson.md`
    - `hardware_notes/intel_openvino.md`

### Chapter 3 — Inference basics: from checkpoint to prediction

- **Goal:** Re-train the deployment mindset: load a checkpoint and run inference cleanly.
- **Topics:** Checkpoint loading; `model.eval()`; `torch.no_grad()`; preprocessing consistency; single-image vs batch; warm-up and basic latency.
- **Outcomes:** Load a trained PyTorch model, run inference on a single image and on a folder, and measure basic inference time.
- **Files:**
    - `docs/03_model_inference_basics.md`
    - `src/inference/infer_pytorch.py` (CLI: `python src/inference/infer_pytorch.py --image datasets/sample.jpg`)
    - `notebooks/chapter_01_latency_benchmarking.ipynb`

### Chapter 4 — Benchmarking: latency, FPS, memory, throughput

- **Goal:** Measure performance honestly. P50 / P95 / P99, end-to-end vs model-only, peak RAM and VRAM.
- **Topics:** Mean latency vs percentiles; throughput vs FPS; warm-up iterations; memory usage; CPU / GPU / NPU utilization; model-only vs end-to-end latency.
- **Outcomes:** Use the course's benchmark library to produce a standard report; distinguish inference time from total pipeline time; save JSON to `experiments/benchmark_results/`.
- **Files:**
    - `docs/04_benchmarking_and_profiling.md`
    - `src/benchmark/__init__.py`, `latency.py`, `memory.py`, `fps.py`, `report.py`, `__main__.py`
    - CLI: `python -m src.benchmark --model mobilenet_v3_small --device cpu --pipeline`
    - `assignments/assignment_03_latency_benchmark.md` (template + 100-point rubric)

---

## Part 2 — Model Deployment

### Chapter 5 — Export model: PyTorch -> ONNX -> Runtime

- **Goal:** Understand ONNX as the intermediate format that connects training to many runtimes.
- **Topics:** Why export; TorchScript; ONNX; static vs dynamic shape; opset; validating; comparing PyTorch and ONNX outputs; common export errors.
- **Outcomes:** Export a PyTorch model to ONNX (single file, `dynamo=False` path); verify with `onnx.checker`; compare outputs against PyTorch within tolerance.
- **Files:**
    - `docs/05_pytorch_to_onnx.md`
    - `src/export/export_onnx.py` (CLI: `python src/export/export_onnx.py --model mobilenet_v3_small`)
    - `notebooks/chapter_02_pytorch_to_onnx.ipynb`
    - `labs/lab_02_export_pytorch_to_onnx.ipynb`

### Chapter 6 — ONNX Runtime for edge inference

- **Goal:** Run models via ONNX Runtime and understand execution providers.
- **Topics:** CPUExecutionProvider, CUDAExecutionProvider, TensorrtExecutionProvider, OpenVINOExecutionProvider; session options; input / output binding; benchmarking ORT.
- **Outcomes:** Load an ONNX model in ONNX Runtime, run inference on CPU or GPU, swap providers when supported, and benchmark using `src.benchmark`.
- **Files:**
    - `docs/06_onnx_runtime.md`
    - `src/inference/infer_onnxruntime.py`
    - `notebooks/chapter_03_onnxruntime_inference.ipynb`

### Chapter 7 — TensorRT, OpenVINO, TFLite

- **Goal:** Survey the optimized runtimes used in production deployment, plus a decision tree.
- **Topics:** TensorRT for NVIDIA GPU / Jetson; OpenVINO for Intel hardware; TFLite for mobile / edge; TensorFlow Lite Micro for microcontroller; FP16 / INT8; calibration data; when to pick which runtime.
- **Outcomes:** Explain the role of an inference engine; describe the PyTorch -> ONNX -> TensorRT / OpenVINO workflow and the TensorFlow / Keras -> TFLite workflow; choose the right runtime for a target.
- **Files:**
    - `docs/07_tensorrt_openvino_tflite.md`
    - `src/inference/infer_tensorrt.py` (template — needs TensorRT installed on the device)
    - `src/inference/infer_openvino.py` (template — runs on Intel CPU at minimum)
    - `labs/lab_04_tensorrt_or_openvino.md` (two tracks: Jetson or Intel)

### Chapter 8 — Model optimization: quantization, pruning, distillation

- **Goal:** Shrink and speed up models while controlling accuracy loss.
- **Topics:** FP32 / FP16 / INT8 / INT4; dynamic vs static quantization; PTQ; intro QAT; structured / unstructured pruning; knowledge distillation; accuracy-vs-latency trade-off.
- **Outcomes:** Quantize a small model end-to-end (dynamic + static); compare model size, latency, and accuracy vs FP32 using `src.optimization.compare_models`; explain why small CNNs sometimes lose latency from CPU INT8.
- **Files:**
    - `docs/08_model_optimization.md`
    - `src/optimization/__init__.py`, `quantization.py`
    - `notebooks/chapter_04_quantization_ptq.ipynb`
    - `assignments/assignment_04_quantization.md` (template + 100-point rubric)

---

## Part 3 — Edge Applications

### Chapter 9 — Real-time camera AI with OpenCV

- **Goal:** Build a real-time inference loop on a webcam (or video file).
- **Topics:** OpenCV VideoCapture; frame loop; resize / normalize; inference and postprocessing; overlay; FPS counter; frame skipping; threading; recording.
- **Outcomes:** Write a real-time camera classifier; display FPS; optimize the pipeline via resize / frame skip / model swap; record an annotated demo video.
- **Files:**
    - `docs/09_realtime_camera_ai.md`
    - `src/inference/camera_loop.py` (CLI with `--source`, `--model`, `--no-display`, `--record`, `--skip`, `--max-frames`)
    - `notebooks/chapter_06_camera_inference_opencv.ipynb`
    - `labs/lab_05_realtime_camera.md`
    - `projects/project_01_camera_classifier/README.md`

### Chapter 10 — Object detection on the edge

- **Goal:** Move from classification to YOLO-style detection for real camera AI.
- **Topics:** Bounding boxes; confidence score; NMS; YOLO family (hands-on); lightweight detectors; input resolution vs FPS; basic tracking; use cases (people counting, PPE detection, defect detection).
- **Outcomes:** Run YOLO on an image, video, and webcam; export YOLO to ONNX; benchmark FPS; tune confidence and IoU thresholds.
- **Files:**
    - `docs/10_object_detection_edge.md`
    - `notebooks/chapter_07_yolo_edge_detection.ipynb`
    - `projects/project_02_yolo_realtime_camera/README.md`, `config.yaml`, `run_image.py`, `run_video.py`, `export_onnx.py`

### Chapter 11 — Edge AI for sensors and time-series

- **Goal:** Extend edge AI beyond camera to industrial sensors, IoT, and wearables.
- **Topics:** Sensor data; sliding-window features; feature extraction; anomaly detection; vibration / temperature / current / pressure sensors; tiny models; basic MQTT.
- **Outcomes:** Window sensor data; train an autoencoder anomaly detector; deploy an inference loop on a simulated stream; emit alerts JSON (MQTT-shaped).
- **Files:**
    - `docs/11_sensor_ai_timeseries.md`
    - `notebooks/chapter_08_sensor_anomaly_detection.ipynb`
    - `projects/project_03_sensor_anomaly_detection/README.md`, `config.yaml`, `generate_synthetic.py`, `train_autoencoder.py`, `infer_stream.py`

### Chapter 12 — TinyML and microcontroller AI

- **Goal:** Introduce TinyML for ultra-constrained devices.
- **Topics:** What TinyML is; microcontroller vs Pi / Jetson; TensorFlow Lite Micro; Edge Impulse workflow (orientation); keyword spotting; gesture recognition; memory constraints; mandatory INT8 quantization.
- **Outcomes:** Identify when TinyML is the right choice; understand the train -> quantize -> deploy workflow; estimate model + activation RAM and decide which boards it fits.
- **Files:**
    - `docs/12_tinyml_microcontrollers.md`
    - `notebooks/chapter_09_tinyml_intro.ipynb`
    - `projects/project_05_tinyml_keyword_spotting/README.md`, `config.yaml`, `generate_synthetic.py`, `train_kws.py`, `footprint.py`

### Chapter 13 — Industrial quality inspection

- **Goal:** Design a near-market use case: detecting product defects on a moving line with camera AI.
- **Topics:** Visual inspection; defect classification / detection / segmentation; lighting and camera placement; false positive / false negative cost; dataset collection (MVTec AD as benchmark); edge deployment at the factory line; human-in-the-loop review.
- **Outcomes:** Design a defect-inspection pipeline; understand industrial data issues (lack of defective samples, class imbalance, lighting variation); pick the right metric.
- **Files:**
    - `docs/13_industrial_quality_inspection.md`
    - `projects/project_04_quality_inspection_ai/README.md` (concept skeleton with recommended datasets and modeling guidance)

---

## Part 4 — Physical AI

### Chapter 14 — Physical AI: perception -> state -> decision -> action

- **Goal:** Frame Physical AI as a closed action loop.
- **Topics:** What Physical AI is; robot perception; world / state representation; decision layer; action layer; control loop; actuator; closed-loop system; rule-based vs learned policy; safety layer.
- **Outcomes:** Describe the sensors -> perception -> state -> decision -> controller -> actuator -> environment pipeline; distinguish perception-only AI from Physical AI.
- **Files:**
    - `docs/14_physical_ai_loop.md`
    - `src/physical_ai/__init__.py`, `perception.py`, `decision.py`, `controller.py`

### Chapter 15 — Control loop, safety layer, and feedback

- **Goal:** Introduce control and safety fundamentals for Physical AI.
- **Topics:** Open-loop vs closed-loop; controller basics; PID (concept); action command; safety boundary; human override; fallback logic; timeout and emergency stop in the simulator.
- **Outcomes:** Design a simple controller in simulation; insert a safety gate before action commands; log action and failure states.
- **Files:**
    - `docs/15_control_safety_feedback.md`
    - `src/physical_ai/safety.py`
    - `labs/lab_08_robot_simulation_loop.md`

### Chapter 16 — Simulation, robotics, and sim-to-real

- **Goal:** Introduce simulation as the necessary step before deploying Physical AI to the real world.
- **Topics:** Why simulate; synthetic data; domain randomization; sim-to-real gap; reinforcement learning (orientation); ROS / ROS2 (intro); NVIDIA Isaac / Gazebo / PyBullet (concept).
- **Outcomes:** Explain why robots cannot be trained or tested entirely in the real world; describe sim-to-real; build a toy simulation loop in Python.
- **Files:**
    - `docs/16_simulation_sim_to_real.md`
    - `notebooks/chapter_10_physical_ai_control_loop.ipynb`
    - `projects/project_06_physical_ai_simulation/README.md`, `config.yaml`, `world.py`, `run_simulation.py`

### Chapter 17 — ROS2 / robot simulator intro

- **Goal:** Bridge from Physical AI fundamentals to robotics engineering.
- **Topics:** ROS2 concepts (node, topic, message, service); publisher / subscriber; camera / sensor topics; command velocity / action topics; simulator integration patterns. ROS2 is not strictly required for the core course.
- **Outcomes:** Understand how an AI module talks to a robot stack; design a mock ROS-like pub/sub in Python where ROS2 is not installed.
- **Files:**
    - `docs/17_ros2_robot_sim_intro.md`
    - `labs/lab_09_mock_ros_loop.md`

---

## Part 5 — Advanced Topics and Final Project

### Chapter 18 — Edge LLM and multimodal AI on devices

- **Goal:** Survey on-device LLM and VLM trends; positioned as advanced (not required for newcomers).
- **Topics:** On-device LLM; small language models; quantized LLM (INT4); vision-language models at edge; prompt latency (TTFT) and tokens / second; memory footprint and KV cache; local privacy; robot instruction interface (VLA orientation); practical limits.
- **Outcomes:** Decide when to run a small local model vs call a cloud LLM; sketch a camera -> vision model -> local LLM decision pipeline; estimate which model fits which device.
- **Files:**
    - `docs/18_edge_llm_multimodal_ai.md`
    - `notebooks/chapter_11_edge_llm_intro.ipynb`

### Chapter 19 — Security, reliability, monitoring, and EdgeOps

- **Goal:** Teach how to operate an Edge AI system after the demo works.
- **Topics:** Model failure; sensor failure; data drift; privacy; secure update; logging; latency / temperature / error monitoring; model versioning; rollback; fallback logic.
- **Outcomes:** Write a risk checklist; design a fallback strategy; define a device log schema; produce a model checksum + version record.
- **Files:**
    - `docs/19_security_reliability_edgeops.md`
    - `src/edgeops/__init__.py`, `logging.py`, `versioning.py`, `fallback.py`
    - `deployment_notes/runtime.md` (template for per-project use)
    - `deployment_notes/safety.md` (template for per-project use)

### Chapter 20 — Final project end-to-end

- **Goal:** Demonstrate end-to-end deployment competence on a chosen problem.
- **Topics:** Problem statement; device / simulator environment; dataset; baseline model; export / optimization; runtime; benchmark; metric; demo script; error analysis; deployment notes; risk / safety notes.
- **Outcomes:** A reproducible repo project with benchmark, demo, deployment notes, and safety notes; graded by a 100-point rubric.
- **Files:**
    - `docs/20_final_project.md`
    - `projects/final_project_template/README.md`
    - `reports/final_project_report_template.md` (the report template the rubric grades against)

---

## Suggested project ideas for the final project

- Real-time camera image classifier on a laptop or Jetson.
- YOLO real-time object detection (people counting, PPE / helmet detection, vehicle detection).
- Product defect detection (MVTec AD or your own dataset).
- Sensor anomaly detection edge pipeline.
- TinyML keyword spotting or gesture recognition.
- Edge dashboard for camera AI.
- Physical AI simulator extension (richer perception, learned policy, safety stress test).
- Edge LLM assistant for camera monitoring (bonus / advanced).
