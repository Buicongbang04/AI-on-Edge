# Chapter 13 — Industrial quality inspection

> **Goal:** Design a near-market camera AI use case — detecting product defects on a moving production line. By the end of this chapter you should be able to design a defect-inspection pipeline, understand the data issues unique to industrial settings, and pick the right metric for the cost structure.

This chapter is more **applied** than the previous ones: it does not introduce new model techniques, but it shows how to combine what Chapters 1-12 taught into a deployable factory-floor system.

---

## 1. The use case

A production line moves products past a fixed camera. The system must:

- Acquire one (or several) images per product as it passes.
- Decide: PASS or FAIL (binary classifier) or pinpoint defect locations (detection / segmentation).
- For FAIL: optionally route the product to a reject bin (actuator command — already crossing into Physical AI, Ch 14-17).
- Log every decision with timestamps and the model's confidence.
- Allow a human operator to review a sample of borderline cases (human-in-the-loop).

Examples in industry:

- PCB inspection (missing components, solder bridges).
- Automotive part inspection (scratches, dents, casting defects).
- Food / agriculture (mold, rot, off-color, foreign objects).
- Pharma (capsule shape, fill level, label print quality).
- Textile (weave defects, prints).
- Packaging (label position, barcode legibility, seal integrity).

The pipeline shape is the same; the model class differs by *what* the defect looks like.

---

## 2. Model choices by defect type

| Defect type | Model class | Why |
|---|---|---|
| Whole-product PASS/FAIL | Classifier (MobileNet, EfficientNet) | Simple, fast, easy to label |
| Localized defects (scratches, dents) | Object detector (YOLO, RetinaNet) | Reports box per defect; usable for routing |
| Pixel-level defects (stains, weave faults) | Segmentation (U-Net, Mask R-CNN) | Some defects are amorphous; bounding boxes are unhelpful |
| Texture anomalies | Autoencoder reconstruction error | When defect examples are too rare to label |
| Anomaly detection in feature space | PaDiM, PatchCore (anomaly detection on features) | When you have lots of "normal" but few "defective" examples — common in industry |

Industrial datasets are often **highly imbalanced**: 1 defective product per 1000+ normal ones. Anomaly detection often beats classification because labels are scarce.

**MVTec AD** (mvtec.com/company/research/datasets/mvtec-ad) is the standard public benchmark for industrial anomaly detection and is the recommended dataset for the project.

---

## 3. Data: the hardest part

A defect-inspection project usually does **not** fail at modeling. It fails at data:

| Problem | Symptom | Mitigation |
|---|---|---|
| Too few defective samples | Model "memorizes" the few defects, fails on new ones | Anomaly detection on normal data only; aggressive augmentation; synthetic defect generation |
| Lighting changes (day / night, lamp aging) | Accuracy collapses after a few weeks | Diffuse / controlled lighting; periodic re-calibration; drift monitoring |
| Camera angle drift | False positives spike when the camera shifts 1° | Mechanical mounting; image registration / homography correction |
| Class imbalance | 99.9% accuracy meaningless | Use recall on defective; cost-weighted metrics |
| New product variants | Old defects no longer apply | Continuous learning loop; periodic re-validation |
| "Pseudo-defects" | Operator-acceptable variations look like defects | Add a "tolerable" class; rely on human-in-the-loop review |

**Spend the first month of any industrial project on data collection, labeling rules, and lighting** — not on the model. This is the highest-leverage advice for the chapter.

---

## 4. Lighting and camera placement

The single highest-impact engineering decision for any vision-based inspection is **lighting**:

- **Diffuse lighting** (light tent, ring light) — reduces shadows and highlights; good for surface defects.
- **Coaxial lighting** — for shiny / mirror surfaces; eliminates direct reflections.
- **Backlight** — for shape / silhouette inspection.
- **UV light** — for inks, plastics, organic contamination.
- **Polarized light** — for glass, screens.

Cameras:

- **Industrial GigE / USB3 cameras** (Basler, FLIR, Sony) — recommended for any production deployment. Higher frame rate, stable settings, hardware trigger.
- **MIPI CSI cameras** (Jetson, Pi) — OK for prototypes; not for harsh factory conditions.
- **USB webcams** — only for the lab.

For triggered acquisition: when the product passes a photoelectric sensor, fire the camera once. This gives reproducible exposure and avoids motion blur.

---

## 5. The metric

Industrial inspection has a non-symmetric cost structure:

- **False negative** (missed defect): the defective product ships → customer complaint, recall, brand damage.
- **False positive** (good product flagged): wasted unit, possibly re-inspected by a human.

Almost always FN cost >> FP cost. The metric reflects that:

- **Recall on the defective class** is the primary KPI.
- **Precision** is the secondary KPI; below some threshold the system is "the boy who cried wolf" and operators ignore it.
- **F1** is a compromise; report it but do not optimize on it alone.
- **Cost per error** (in $) is the deployment KPI: weight FN by recall miss × cost-of-recall, FP by precision miss × cost-of-rejection.

Set a **target recall ≥ 0.95** for safety-critical defects (medical, automotive). Set a **target precision** floor (often ≥ 0.7-0.8) so operators trust the system.

---

## 6. Edge deployment on the line

Typical hardware:

- **Industrial PC + Intel iGPU** (NUC, IEI, Advantech) + OpenVINO. Robust, IT-friendly, supports multiple cameras.
- **NVIDIA Jetson Orin Nano / Orin NX** + TensorRT. Best perf/$ for multi-camera CV.
- **Smart camera** (Cognex In-Sight, Basler Pulse) with built-in inference. Closed ecosystem, but factory-grade.

Connectivity:

- **PLC integration via MQTT or OPC-UA** for actuator commands (reject arm, conveyor stop).
- **Web dashboard** for operator review.
- **MES / ERP integration** for batch tracking.

Latency budget: usually driven by conveyor belt speed and product spacing. Example: a 0.5 m/s belt with 5 cm spacing leaves 100 ms per product (capture + inference + decision + actuator command). Plan accordingly (Ch 1 latency budget).

---

## 7. Human-in-the-loop review

A defect-inspection system should **not** make autonomous decisions for all units in its first month. Common rollout:

1. **Shadow mode** — model runs but operator decides; logs are compared after the fact.
2. **Borderline-only mode** — model decides only for high-confidence cases; sends low-confidence to operator.
3. **Autonomous mode** — model decides for everything; operator audits a random 1-5% sample.

Even in autonomous mode, *always* sample for human review. This is how you catch drift before the customer does.

---

## 8. The worked project: `projects/project_04_quality_inspection_ai/`

The project is a *concept skeleton*. It does not ship a real defect dataset (those are usually proprietary or require download from MVTec AD), but it provides:

- README with the full inspection pipeline.
- Suggested dataset choices (MVTec AD subsets, your own).
- A baseline training script template for both:
    - Supervised classifier (PASS/FAIL).
    - Anomaly detection (PaDiM / autoencoder, building on Chapter 11).
- Configuration for thresholding and operating point selection.

For the final project (Ch 20), this is one of the suggested templates.

---

## 9. Common pitfalls

- **Optimizing accuracy on an imbalanced dataset.** Use recall + precision (with cost weighting). 99.5% accuracy on a 1% defect rate is the same as guessing "PASS" every time.
- **Not splitting by time or by lot.** Random splits leak across the same lighting / product batch and overestimate performance.
- **Training only on the obvious defects.** Real defective products show partial, ambiguous, atypical defects. Collect those too.
- **Ignoring drift.** A model that works in week 1 may degrade by week 4 because lamps dim, dust accumulates, parts subtly change.
- **No fallback.** When confidence is low or the camera disconnects, the system needs to default to *human review*, not to a silent PASS.
- **Operator alert fatigue.** If precision is too low, operators stop trusting alerts. Tune precision floor jointly with recall target.

---

## 10. What you should be able to do after this chapter

- Design the data and lighting plan for a defect-inspection use case (the first month of work).
- Pick model class (classifier / detector / segmenter / anomaly detector) based on the defect type and label availability.
- Set realistic recall / precision targets driven by the cost ratio.
- Sketch the line-side deployment: hardware, runtime, PLC integration, dashboard, human-in-the-loop.
- Plan for drift, fallback, and monitoring from day one (Chapter 19 builds on this).

---

## 11. Files produced by this chapter

- `docs/13_industrial_quality_inspection.md` — this file.
- `projects/project_04_quality_inspection_ai/` — project skeleton + README (suggested datasets, baseline structure, dashboard plan).
