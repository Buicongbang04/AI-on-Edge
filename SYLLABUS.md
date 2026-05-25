# Syllabus

Chapter-by-chapter syllabus for **Edge AI and Introductory Physical AI**. Each chapter lists its goal, main topics, learning outcomes, and the files it produces in this repo.

For the 5-part structure, see [`COURSE_OVERVIEW.md`](COURSE_OVERVIEW.md). For the week-by-week plan, see [`ROADMAP.md`](ROADMAP.md).

---

## Part 1 — Edge AI Foundation

### Chapter 0 — Edge AI and Physical AI overview
- **Goal:** Distinguish Edge AI vs Cloud AI, define Physical AI, and motivate why this matters.
- **Topics:** Cloud AI vs Edge AI; on-device inference; real-time AI; latency / throughput / FPS; privacy & offline inference; Physical AI (perception, state, decision, action); use cases.
- **Outcomes:** Explain Edge AI vs Physical AI in your own words; distinguish from Embedded AI / TinyML / Robotics AI; cite real-world examples per application group.
- **Artifacts:** `docs/00_intro_edge_physical_ai.md`, `figures/cloud_vs_edge.png`, `figures/physical_ai_loop.png`.

### Chapter 1 — Edge AI system design
- **Goal:** Frame an Edge AI problem as a system, not a model choice.
- **Topics:** Input source (camera/mic/sensor/video); preprocessing/inference/postprocessing; output (label/box/mask/command/alert); latency budget; accuracy requirement; hardware/power/deployment environment; failure modes.
- **Outcomes:** Write a short system-design note for a given Edge AI application; specify input, output, metric, latency target, and deployment risks.
- **Artifacts:** `docs/01_edge_ai_system_design.md`, `assignments/assignment_01_edge_ai_analysis.md`.

### Chapter 2 — Hardware for Edge AI
- **Goal:** Map application classes to hardware classes.
- **Topics:** CPU / GPU / NPU / TPU accelerators; microcontroller vs Raspberry Pi vs NVIDIA Jetson vs Intel CPU/NPU vs Google Coral TPU; camera modules and sensors; cost / power / compute / ecosystem trade-offs.
- **Outcomes:** Tell the difference between microcontroller, SBC, and edge GPU; pick the right device class for a given task.
- **Artifacts:** `docs/02_hardware_for_edge_ai.md`, `hardware_notes/{laptop_cpu_gpu, raspberry_pi, nvidia_jetson, intel_openvino}.md`.

### Chapter 3 — Inference basics: from checkpoint to prediction
- **Goal:** Re-train the deployment mindset: load a checkpoint and run inference cleanly.
- **Topics:** Checkpoint loading; `model.eval()`; `torch.no_grad()`; preprocessing consistency; single-image vs batch inference; warm-up and basic latency measurement.
- **Outcomes:** Load a trained PyTorch model, run inference on a single image and on a folder, and measure basic inference time.
- **Artifacts:** `notebooks/chapter_01_latency_benchmarking.ipynb`, `src/inference/infer_pytorch.py`, `docs/03_model_inference_basics.md`.

### Chapter 4 — Benchmarking: Latency, FPS, Memory, Throughput
- **Goal:** Measure performance honestly.
- **Topics:** Mean latency vs P50/P95/P99; throughput vs FPS; warm-up iterations; memory usage; CPU/GPU/NPU utilization; model-only latency vs end-to-end latency.
- **Outcomes:** Write a benchmark script that reports latency percentiles; distinguish inference time from total pipeline time; produce a benchmark table.
- **Artifacts:** `src/benchmark/{latency, memory, fps}.py`, `docs/04_benchmarking_and_profiling.md`, `assignments/assignment_03_latency_benchmark.md`.

---

## Part 2 — Model Deployment

### Chapter 5 — Export model: PyTorch → ONNX → Runtime
- **Goal:** Understand ONNX as the intermediate format that connects training to many runtimes.
- **Topics:** Why export; TorchScript; ONNX; static vs dynamic shape; opset; validating exported models; comparing PyTorch and ONNX outputs; common export errors.
- **Outcomes:** Export a PyTorch model to ONNX; verify it with ONNX Runtime; compare outputs.
- **Artifacts:** `notebooks/chapter_02_pytorch_to_onnx.ipynb`, `labs/lab_02_export_pytorch_to_onnx.ipynb`, `src/export/export_onnx.py`, `docs/05_pytorch_to_onnx.md`.

### Chapter 6 — ONNX Runtime for edge inference
- **Goal:** Run models via ONNX Runtime and understand execution providers.
- **Topics:** CPUExecutionProvider, CUDAExecutionProvider, TensorRTExecutionProvider (concept), OpenVINOExecutionProvider (concept); session options; input/output binding; benchmarking ONNX Runtime.
- **Outcomes:** Load an ONNX model in ONNX Runtime, run inference on CPU or GPU, and swap execution providers when supported.
- **Artifacts:** `notebooks/chapter_03_onnxruntime_inference.ipynb`, `src/inference/infer_onnxruntime.py`, `docs/06_onnx_runtime.md`.

### Chapter 7 — TensorRT, OpenVINO, TFLite
- **Goal:** Survey the optimized runtimes used in production deployment.
- **Topics:** TensorRT for NVIDIA GPUs / Jetson; OpenVINO for Intel hardware; TFLite for mobile / edge; TensorFlow Lite Micro for microcontrollers; FP16 / INT8 inference; calibration data; when to pick which runtime.
- **Outcomes:** Explain the role of an inference engine; describe the PyTorch → ONNX → TensorRT/OpenVINO workflow and the TensorFlow / Keras → TFLite workflow.
- **Artifacts:** `labs/lab_04_tensorrt_or_openvino.md`, `src/inference/{infer_tensorrt, infer_openvino}.py`, `docs/07_tensorrt_openvino_tflite.md`.

### Chapter 8 — Model optimization: quantization, pruning, distillation
- **Goal:** Shrink and speed up models while controlling accuracy loss.
- **Topics:** FP32 / FP16 / INT8; dynamic vs static quantization; post-training quantization; intro to quantization-aware training; structured / unstructured pruning; knowledge distillation; accuracy-vs-latency trade-off.
- **Outcomes:** Quantize a small model end-to-end; compare model size, latency, and accuracy before/after; explain when quantization hurts accuracy.
- **Artifacts:** `notebooks/chapter_04_quantization_ptq.ipynb`, `src/optimization/quantization.py`, `assignments/assignment_04_quantization.md`, `docs/08_model_optimization.md`.

---

## Part 3 — Edge Applications

### Chapter 9 — Real-time camera AI with OpenCV
- **Goal:** Build a real-time inference pipeline using a webcam.
- **Topics:** OpenCV VideoCapture; camera frame loop; resize / normalize; inference and postprocessing; overlay; FPS counter; frame skipping; basic threading.
- **Outcomes:** Write a real-time camera classifier; display FPS; optimize the pipeline via resize, frame skip, or a smaller model.
- **Artifacts:** `notebooks/chapter_06_camera_inference_opencv.ipynb`, `src/inference/camera_loop.py`, `labs/lab_05_realtime_camera.md`, `docs/09_realtime_camera_ai.md`.

### Chapter 10 — Object detection on the edge
- **Goal:** Move from classification to YOLO-style detection for real camera AI.
- **Topics:** Bounding boxes; confidence score; NMS; YOLO family (hands-on); lightweight detectors; input resolution vs FPS; basic tracking; use cases (people counting, product detection, safety helmet).
- **Outcomes:** Run YOLO on an image or video; export YOLO to ONNX; benchmark FPS; tune the confidence threshold.
- **Artifacts:** `projects/project_02_yolo_realtime_camera/`, `notebooks/chapter_07_yolo_edge_detection.ipynb`, `docs/10_object_detection_edge.md`.

### Chapter 11 — Edge AI for sensors and time-series
- **Goal:** Extend edge AI beyond camera to industrial sensors, IoT, and wearables.
- **Topics:** Sensor data; sliding-window features; feature extraction; anomaly detection; vibration / temperature / current / pressure sensors; tiny models for sensors; basic MQTT.
- **Outcomes:** Window sensor data; train an anomaly detection model; deploy an inference loop on a simulated stream; emit alerts via MQTT or log dashboard.
- **Artifacts:** `projects/project_03_sensor_anomaly_detection/`, `notebooks/chapter_08_sensor_anomaly_detection.ipynb`, `docs/11_sensor_ai_timeseries.md`.

### Chapter 12 — TinyML and microcontroller AI
- **Goal:** Introduce TinyML for ultra-constrained devices.
- **Topics:** What TinyML is; microcontroller vs Raspberry Pi / Jetson; TensorFlow Lite Micro; Edge Impulse workflow (concept); keyword spotting; gesture recognition; memory constraints; mandatory quantization.
- **Outcomes:** Identify when TinyML is the right choice; understand the train → quantize → deploy workflow; run a keyword-spotting / sensor classification demo where boards are available.
- **Artifacts:** `projects/project_05_tinyml_keyword_spotting/`, `notebooks/chapter_09_tinyml_intro.ipynb`, `docs/12_tinyml_microcontrollers.md`.

### Chapter 13 — Industrial quality inspection
- **Goal:** Design a near-market use case: detecting product defects with camera AI.
- **Topics:** Visual inspection; defect classification / detection / segmentation; lighting and camera placement; false positive / false negative cost; dataset collection; edge deployment at the factory line; human-in-the-loop review.
- **Outcomes:** Design a defect-inspection pipeline; understand industrial data issues (lack of defective samples, class imbalance, lighting variation); pick the right metric.
- **Artifacts:** `projects/project_04_quality_inspection_ai/`, `docs/13_industrial_quality_inspection.md`.

---

## Part 4 — Physical AI

### Chapter 14 — Physical AI: perception → state → decision → action
- **Goal:** Frame Physical AI as a closed action loop.
- **Topics:** What Physical AI is; robot perception; world / state representation; decision layer; action layer; control loop; actuator; closed-loop system; rule-based decision vs learned policy; safety layer.
- **Outcomes:** Describe the sensors → perception → state → decision → controller → actuator → environment pipeline; distinguish perception-only AI from Physical AI.
- **Artifacts:** `src/physical_ai/{perception, decision, controller}.py`, `docs/14_physical_ai_loop.md`.

### Chapter 15 — Control loop, safety layer, and feedback
- **Goal:** Introduce control and safety fundamentals for Physical AI.
- **Topics:** Open-loop vs closed-loop; basic controller; PID (concept); action command; safety boundary; human override; fallback logic; timeout and emergency stop in simulators.
- **Outcomes:** Design a simple controller in simulation; insert a safety gate before action commands; log action and failure states.
- **Artifacts:** `labs/lab_08_robot_simulation_loop.md`, `src/physical_ai/{controller, safety}.py`, `docs/15_control_safety_feedback.md`.

### Chapter 16 — Simulation, robotics, and sim-to-real
- **Goal:** Introduce simulation as the necessary step before deploying Physical AI to the real world.
- **Topics:** Why simulate; synthetic data; domain randomization; sim-to-real gap; reinforcement learning (orientation level only); ROS / ROS2 (intro); NVIDIA Isaac / Gazebo / PyBullet (concept).
- **Outcomes:** Explain why robots cannot be trained or tested entirely in the real world; describe sim-to-real; build a toy Python simulation loop.
- **Artifacts:** `projects/project_06_physical_ai_simulation/`, `notebooks/chapter_10_physical_ai_control_loop.ipynb`, `docs/16_simulation_sim_to_real.md`.

### Chapter 17 — ROS2 / robot simulator intro
- **Goal:** Bridge from Physical AI fundamentals to robotics engineering.
- **Topics:** ROS2 concepts (node, topic, message, service — orientation level); publisher / subscriber; camera / sensor topics; command velocity / action topics; simulator integration patterns; ROS2 not strictly required for the core course.
- **Outcomes:** Understand how an AI module talks to a robot stack; design a mock ROS-like pub/sub in Python where ROS2 is not installed.
- **Artifacts:** `labs/lab_09_mock_ros_loop.md`, `docs/17_ros2_robot_sim_intro.md`.

---

## Part 5 — Advanced Topics and Final Project

### Chapter 18 — Edge LLM and multimodal AI on devices
- **Goal:** Survey on-device LLM / VLM trends, positioned as advanced (not required for newcomers).
- **Topics:** On-device LLM; small language models; quantized LLM; vision-language models on edge; prompt latency; memory footprint; local privacy; robot instruction interface; practical limits of edge LLM.
- **Outcomes:** Decide when to run a small local model vs call a cloud LLM; sketch a camera → vision model → local LLM decision pipeline.
- **Artifacts:** `notebooks/chapter_11_edge_llm_intro.ipynb`, `docs/18_edge_llm_multimodal_ai.md`.

### Chapter 19 — Security, reliability, monitoring, and EdgeOps
- **Goal:** Teach how to operate an Edge AI system once it works in demo.
- **Topics:** Model failure; sensor failure; data drift; privacy; secure update; logging; latency / temperature / error monitoring; model versioning; rollback; fallback logic.
- **Outcomes:** Write a risk checklist for an Edge AI system; propose a fallback when the model is uncertain; design a device log schema.
- **Artifacts:** `docs/19_security_reliability_edgeops.md`, `deployment_notes/{runtime, safety}.md`, `src/edgeops/`.

### Chapter 20 — Final project end-to-end
- **Goal:** Demonstrate end-to-end deployment competence on a chosen problem.
- **Topics:** Problem statement; device or simulator environment; dataset; baseline model; export and optimization; runtime; benchmark; metric; demo script; error analysis; deployment notes; risk and safety notes.
- **Outcomes:** A reproducible repo project with benchmark, demo, deployment notes, and safety notes.
- **Artifacts:** `projects/final_project_template/`, `reports/final_project_report_template.md`, `docs/20_final_project.md`.

---

## Suggested project ideas for the final project

- Real-time camera image classifier on laptop or Jetson.
- YOLO real-time object detection (people counting, helmet detection, etc.).
- Product defect detection demo.
- Sensor anomaly detection edge pipeline.
- TinyML keyword spotting or gesture recognition.
- Edge dashboard for camera AI.
- Physical AI simulator: perception → state → decision → action with safety.
- Edge LLM assistant for camera monitoring (bonus).
