# Final project — <your project name>

Template for the course's Chapter 20 capstone project. **Fork this folder** into `projects/<your_project_name>/` and fill it in.

The full rubric and the 12 required sections live in `docs/20_final_project.md` and in the report template at `reports/final_project_report_template.md`.

---

## Quick checklist

- [ ] Pick a problem (see suggested templates in `docs/20_final_project.md`).
- [ ] Fork this folder to `projects/<your_project_name>/`.
- [ ] Fill in `config.yaml`.
- [ ] Write `train.py` (or link to a notebook that does training).
- [ ] Write `export.py` (PyTorch → ONNX, optionally TFLite / TensorRT).
- [ ] Write `infer.py` (camera, video, or stream loop).
- [ ] Write `benchmark.py` (uses `src.benchmark.bench_full`).
- [ ] Run the benchmark, save JSON to `experiments/benchmark_results/`.
- [ ] Record a `results/demo.mp4` or screenshot.
- [ ] Fill in `deployment_notes/runtime.md` and `deployment_notes/safety.md`.
- [ ] Write the report at `reports/<your_project_name>_report.md`.

---

## Files

```
final_project_template/
├── README.md             — this file
├── config.yaml           — runtime config
├── train.py              — model training (or link to notebook)
├── export.py             — model → ONNX / TFLite
├── infer.py              — inference loop
├── benchmark.py          — calls src.benchmark
├── deployment_notes/
│   ├── runtime.md        — see template in repo root deployment_notes/
│   └── safety.md         — see template in repo root deployment_notes/
└── results/              — benchmark JSON, demo video, screenshots
```

---

## Starter `config.yaml`

```yaml
project_name: <name>
problem_statement: >
  <one paragraph: who, what, where, why, with what consequence>

hardware:
  device: <e.g. laptop CPU / Jetson Orin Nano / Raspberry Pi 5 + Coral>
  runtime: <e.g. onnxruntime / tensorrt / openvino / tflite>
  execution_provider: <e.g. CPUExecutionProvider>

model:
  architecture: <e.g. MobileNetV3-Small>
  source_weights: <e.g. torchvision IMAGENET1K_V1 / fine-tuned weights path>
  exported_artifact: experiments/exported_models/<name>.onnx
  precision: <FP32 / FP16 / INT8>
  input_shape: [1, 3, 224, 224]
  preprocessing:
    mean: [0.485, 0.456, 0.406]
    std:  [0.229, 0.224, 0.225]
    resize: 232
    crop:   224

metric:
  primary_quality: <e.g. recall on defect ≥ 0.95>
  primary_performance: <e.g. P95 ≤ 50 ms; FPS ≥ 20>
  cost_tradeoff: <e.g. FN cost >> FP cost>

operating_point:
  confidence_threshold: 0.5
  fallback_strategy: <use_prediction / rule_based / human_review>
```

---

## Notes on what to write

- **README.md (this file)** — explains how to run train / export / infer / benchmark.
- **deployment_notes/runtime.md** — copy from `deployment_notes/runtime.md` in the repo root, fill in.
- **deployment_notes/safety.md** — copy from `deployment_notes/safety.md` in the repo root, fill in.
- **results/** — keep at least the benchmark JSON, the demo video / screenshot, and any error-analysis plots.

---

## Reproducibility checklist

- [ ] `pip install -r requirements.txt` from a fresh env produces a working setup.
- [ ] `python train.py` (or notebook) runs and produces a checkpoint.
- [ ] `python export.py` produces the ONNX in `experiments/exported_models/`.
- [ ] `python benchmark.py` produces a JSON in `experiments/benchmark_results/`.
- [ ] `python infer.py --source <sample> --no-display` produces a demo output.
- [ ] All paths in `config.yaml` are relative; no absolute paths from the author's machine.
