# Project 04 — Industrial quality inspection AI

Skeleton for a defect-inspection mini-project for Chapter 13. The course does not ship a real defect dataset (those are usually proprietary or large); instead, this README walks you through the structure with two recommended public datasets.

If you pick this as your final project, fork this folder, fill in the model and the data loaders for your chosen dataset, and produce the benchmark and deployment notes the rubric expects.

---

## Suggested datasets

| Dataset | Class | Size | Why |
|---|---|---|---|
| **MVTec AD** (mvtec.com/company/research/datasets/mvtec-ad) | Industrial anomaly | ~5K train / 1.7K test images, 15 categories | The standard benchmark for industrial anomaly detection. Many tutorials and reference scores. |
| **DAGM 2007** (kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset) | Texture defects | 10 classes, ~17K images | Classical texture defect dataset; supervised. |
| **NEU Surface Defect** (e.g. on Kaggle) | Steel defects | ~1800 images, 6 classes | Small, easy to handle, good for first-pass classifier. |
| **Custom** | Your problem | varies | If you have your own products to inspect, this is by far the most realistic. |

Pick **MVTec AD** if you want to practice anomaly detection (PaDiM, PatchCore, autoencoder). Pick **DAGM 2007** or **NEU Surface Defect** if you want to practice supervised classification.

---

## Suggested pipeline

```
inspection
├── README.md            — this file
├── config.yaml          — model class, thresholds, paths
├── data/                — dataset (download yourself; keep out of git)
├── train.py             — train a classifier or anomaly detector
├── evaluate.py          — confusion matrix + precision/recall + cost
├── infer_line.py        — simulate a production line: 1 image per "product",
│                          decision + simulated PLC alert
└── results/
    ├── model.pt
    ├── confusion_matrix.png
    ├── pr_curve.png
    └── benchmark.json
```

The course's other reference scripts cover the runtime side: `src/inference/`, `src/benchmark/`, `src/optimization/`. Reuse them.

---

## Modeling choice

| If you have... | Model |
|---|---|
| Many labeled defective examples, few classes | Classifier (MobileNetV3-Large / EfficientNet) |
| Localized defects you want a box for | YOLO (project_02 stack, fine-tuned) |
| Pixel-level defects | U-Net or Mask R-CNN |
| Mostly normal data + few defective | **Anomaly detection** (autoencoder reconstruction error, PaDiM, PatchCore). Strongly recommended for MVTec AD. |

For a clean **first try** on MVTec AD: implement the simple **autoencoder** approach from Chapter 11 (`projects/project_03_sensor_anomaly_detection/`) but on image patches. Score every image by reconstruction error.

---

## Suggested config

```yaml
mode: anomaly           # 'anomaly' or 'classifier'
backbone: resnet18      # used for feature-based anomaly methods
input_size: 224
batch_size: 32
epochs: 30
learning_rate: 0.001
anomaly_threshold_quantile: 0.99
target_recall_defective: 0.95
target_precision_floor:   0.80
```

---

## What to report

| Field | Example |
|---|---|
| Dataset | MVTec AD — bottle, capsule, hazelnut |
| Mode | anomaly (autoencoder reconstruction error) |
| Hardware | Jetson Orin Nano + TensorRT INT8 |
| Image size | 224×224 |
| Test recall (defective) | 0.94 |
| Test precision | 0.86 |
| F1 | 0.90 |
| End-to-end FPS (line simulation) | 25 |
| Latency per product (P95) | 38 ms |
| Cost per error | "FN cost = $50 (recall miss) / FP cost = $0.50 (wasted unit)" |
| What you would do next | Add segmentation for explainability |

---

## Deployment notes (the part most people forget)

Drop the following into `deployment_notes/`:

- `hardware.md` — camera model, lens, lighting, mounting, PLC connection.
- `runtime.md` — runtime version, model version, threshold values, retrain cadence.
- `safety.md` — fallback when confidence < threshold (route to human review), failure modes (camera offline, conveyor offline), human override switch, alert deduplication policy.

These three files are required by the final-project rubric (Ch 20).
